# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "more-itertools==10.8.0",
#   "numpy==2.3.2",
#   "tqdm==4.67.1",
#   "llama-cpp-python==0.3.16"
# ]
# ///

"""
A lossless compression utility that leverages a user-provided LLM as the probabilistic model for an arithmetic coder.

The core idea: Use a language model's ability to predict the next token to drive arithmetic coding.
When compressing, the LLM predicts probabilities for each token, which are used to encode the token
efficiently. When decompressing, the same probabilities are used to decode the original sequence.

Copied and adapted from https://github.com/AlexBuz/llama-zip
"""

# Standard imports
import argparse
import base64
import codecs
import signal
import sys

# Library imports
import numpy as np

from llama_cpp import Llama
from more_itertools import consume
from tqdm import tqdm


# Constants for Unicode Private Use Area (PUA) encoding
# Used to encode raw bytes as valid UTF-8 characters for LLM processing
PUA_START = 0xE000

# Arithmetic coding configuration
NUM_STATE_BITS = 64  # Number of bits in the arithmetic coder state
FREQ_SCALE_FACTOR = 1 << 32  # Scale factor for frequency calculations

# Base64 character sets for encoding/decoding
BASE64 = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
BASE64_EQ = BASE64 + b"="


class ArithmeticCoderBase:
    """
    Base class for arithmetic encoding/decoding.
    
    Arithmetic coding compresses data by representing sequences of symbols
    as a single fractional number within [0,1). The key insight is that
    symbols with higher probabilities get assigned larger intervals.
    """
    def __init__(self):
        full_range = 1 << NUM_STATE_BITS
        self.half_range = full_range >> 1
        self.quarter_range = self.half_range >> 1
        self.state_mask = full_range - 1
        # Current interval bounds [low, high]
        self.low = 0
        self.high = self.state_mask

    def update(self, cum_freqs, symbol):
        """
        Update the interval bounds based on the symbol's probability.
        
        Args:
            cum_freqs: Cumulative frequency distribution from the LLM
            symbol: The token/symbol being encoded/decoded
        """
        total = int(cum_freqs[-1])
        range = self.high - self.low + 1
        
        # Calculate new interval bounds based on symbol's probability
        symhigh = int(cum_freqs[symbol])
        self.high = self.low + symhigh * range // total - 1
        symlow = int(cum_freqs[symbol - 1]) if symbol > 0 else 0
        self.low = self.low + symlow * range // total
        
        # Renormalization: prevent interval from becoming too narrow
        while ((self.low ^ self.high) & self.half_range) == 0:
            self.shift()
            self.low = (self.low << 1) & self.state_mask
            self.high = ((self.high << 1) & self.state_mask) | 1
        while (self.low & ~self.high & self.quarter_range) != 0:
            self.underflow()
            self.low = (self.low << 1) ^ self.half_range
            self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1

    def shift(self):
        raise NotImplementedError()

    def underflow(self):
        raise NotImplementedError()


class Encoder(ArithmeticCoderBase):
    """
    Arithmetic encoder that compresses symbols into a binary stream.
    
    Uses the LLM's probability predictions to assign optimal bit lengths
    to each token - frequent tokens get fewer bits, rare tokens get more.
    """
    def __init__(self):
        super().__init__()
        self.encoded_data = bytearray()  # Output compressed bits
        self.bit_index = 8  # Current bit position in the current byte
        self.num_underflow = 0  # Track underflow events for correct bit output

    def get_encoded(self):
        """Return the compressed data as bytes."""
        return self.encoded_data

    def encode_symbol(self, cum_freqs, symbol):
        """Encode a single symbol using its probability distribution."""
        self.update(cum_freqs, symbol)

    def finish(self):
        """Finalize the encoding process."""
        self.append_bit(1)

    def shift(self):
        """Output a bit when the interval bounds converge."""
        bit = self.low >> (NUM_STATE_BITS - 1)
        self.append_bit(bit)
        # Output any pending underflow bits
        for _ in range(self.num_underflow):
            self.append_bit(bit ^ 1)
        self.num_underflow = 0

    def underflow(self):
        """Handle underflow condition by deferring bit output."""
        self.num_underflow += 1

    def append_bit(self, bit):
        """Add a single bit to the output stream."""
        if self.bit_index == 8:
            self.encoded_data.append(0)
            self.bit_index = 0
        self.encoded_data[-1] |= bit << (7 - self.bit_index)
        self.bit_index += 1


class Decoder(ArithmeticCoderBase):
    """
    Arithmetic decoder that reconstructs symbols from a compressed stream.
    
    Uses the same probability distributions as the encoder to reverse
    the compression process and recover the original token sequence.
    """
    def __init__(self, data: bytes):
        super().__init__()
        self.input = data
        self.byte_index = 0
        self.bit_index = 0
        # Initialize the decoder state with the first NUM_STATE_BITS bits
        self.code = sum(
            self.read_code_bit() << i for i in range(NUM_STATE_BITS - 1, -1, -1)
        )

    def decode_symbol(self, cum_freqs):
        """
        Decode the next symbol using the probability distribution.
        
        The decoder finds which symbol interval contains the current
        code value, effectively reversing the encoding process.
        """
        total = int(cum_freqs[-1])
        range = self.high - self.low + 1
        offset = self.code - self.low
        # Find which symbol's interval contains the current code
        value = ((offset + 1) * total - 1) // range
        symbol = np.searchsorted(cum_freqs, value, side="right")
        self.update(cum_freqs, symbol)
        return symbol

    def shift(self):
        """Read the next bit when interval bounds converge."""
        self.code = ((self.code << 1) & self.state_mask) | self.read_code_bit()

    def underflow(self):
        """Handle underflow by adjusting the code value."""
        self.code = (
            (self.code & self.half_range)
            | ((self.code << 1) & (self.state_mask >> 1))
            | self.read_code_bit()
        )

    def read_code_bit(self):
        """Read the next bit from the compressed data stream."""
        if self.byte_index >= len(self.input):
            return 0
        bit = (self.input[self.byte_index] >> (7 - self.bit_index)) & 1
        self.bit_index = (self.bit_index + 1) % 8
        if self.bit_index == 0:
            self.byte_index += 1
        return bit


# UTF-8 processing utilities
# Based on Rust's std::str::Utf8Chunks
class Utf8Chunks:
    """
    Iterator that splits byte sequences into valid UTF-8 chunks and invalid bytes.
    
    This is crucial for handling arbitrary binary data that may contain
    both valid UTF-8 sequences and raw bytes that need special encoding.
    """
    def __init__(self, source: bytes):
        self.source = source

    def __iter__(self):
        return self

    def __next__(self):
        """Find the next chunk of valid UTF-8 or invalid bytes."""
        if not self.source:
            raise StopIteration

        TAG_CONT_U8 = 128  # UTF-8 continuation byte marker

        def safe_get(xs, i):
            """Safely get byte at index, return 0 if out of bounds."""
            try:
                return xs[i]
            except IndexError:
                return 0

        i = 0
        valid_up_to = 0
        
        # Parse UTF-8 byte sequences according to the standard
        while i < len(self.source):
            byte = self.source[i]
            i += 1

            if byte < 0x80:
                # ASCII (1-byte sequence)
                pass
            elif 0xC2 <= byte <= 0xDF:
                # 2-byte sequence
                if safe_get(self.source, i) & 192 != TAG_CONT_U8:
                    break
                i += 1
            elif 0xE0 <= byte <= 0xEF:
                # 3-byte sequence
                next_byte = safe_get(self.source, i)
                if 0xE0 == byte and 0xA0 <= next_byte <= 0xBF:
                    pass
                elif 0xE1 <= byte <= 0xEC and 0x80 <= next_byte <= 0xBF:
                    pass
                elif 0xED == byte and 0x80 <= next_byte <= 0x9F:
                    pass
                elif 0xEE <= byte <= 0xEF and 0x80 <= next_byte <= 0xBF:
                    pass
                else:
                    break
                i += 1
                if safe_get(self.source, i) & 192 != TAG_CONT_U8:
                    break
                i += 1
            elif 0xF0 <= byte <= 0xF4:
                # 4-byte sequence
                next_byte = safe_get(self.source, i)
                if 0xF0 == byte and 0x90 <= next_byte <= 0xBF:
                    pass
                elif 0xF1 <= byte <= 0xF3 and 0x80 <= next_byte <= 0xBF:
                    pass
                elif 0xF4 == byte and 0x80 <= next_byte <= 0x8F:
                    pass
                else:
                    break
                i += 1
                if safe_get(self.source, i) & 192 != TAG_CONT_U8:
                    break
                i += 1
                if safe_get(self.source, i) & 192 != TAG_CONT_U8:
                    break
                i += 1
            else:
                # Invalid UTF-8 start byte
                break

            valid_up_to = i

        # Split the inspected bytes into valid and invalid portions
        inspected, remaining = self.source[:i], self.source[i:]
        self.source = remaining

        valid, invalid = inspected[:valid_up_to], inspected[valid_up_to:]
        return Utf8Chunk(valid, invalid)


class Utf8Chunk:
    def __init__(self, valid: bytes, invalid: bytes):
        self.valid = valid
        self.invalid = invalid


def bytes_to_utf8(data: bytes):
    """
    Convert arbitrary bytes to valid UTF-8 by encoding invalid bytes in PUA.
    
    This allows binary data to be processed by text-based language models.
    Invalid UTF-8 bytes are mapped to Unicode Private Use Area characters.
    """
    output = bytearray()
    for chunk in Utf8Chunks(data):
        # Process valid UTF-8 characters
        for char in chunk.valid.decode("utf-8"):
            # Double-encode PUA characters to avoid conflicts
            if PUA_START <= ord(char) <= PUA_START + 0xFF:
                for byte in char.encode("utf-8"):
                    output.extend(chr(PUA_START + byte).encode("utf-8"))
            else:
                output.extend(char.encode("utf-8"))
        # Encode invalid bytes as PUA characters
        for byte in chunk.invalid:
            output.extend(chr(PUA_START + byte).encode("utf-8"))
    return bytes(output)


def utf8_to_bytes(data: str):
    """
    Convert PUA-encoded UTF-8 back to original bytes.
    
    This reverses the bytes_to_utf8 transformation, recovering
    the original binary data from the LLM-processable text.
    """
    output = bytearray()
    for char in data:
        if PUA_START <= ord(char) <= PUA_START + 0xFF:
            # Convert PUA character back to original byte
            output.append(ord(char) - PUA_START)
        else:
            # Regular UTF-8 character
            output.extend(char.encode("utf-8"))
    return bytes(output)


class LlamaZip:
    """
    Main compression/decompression class using LLaMA models.
    
    This class orchestrates the entire compression process:
    1. Load and configure the language model
    2. Convert data to LLM-compatible format
    3. Use model predictions to drive arithmetic coding
    4. Handle context windows and memory management
    """
    def __init__(
        self, model_path, n_ctx=0, n_gpu_layers=-1, use_mlock=False, verbose=False
    ):
        self.verbose = verbose
        self.load_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            use_mlock=use_mlock,
        )

    def load_model(self, model_path, n_ctx, n_gpu_layers, use_mlock):
        """Load the LLaMA model with specified configuration."""
        loading_message = "Loading model..."
        if self.verbose:
            print(loading_message, end="", flush=True, file=sys.stderr)
        self.model = Llama(
            model_path=model_path,
            use_mlock=use_mlock,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
        if self.verbose:
            print(
                "\r" + " " * len(loading_message) + "\r",
                end="",
                flush=True,
                file=sys.stderr,
            )

    def compute_cdf(self, logits):
        """
        Convert model logits to cumulative distribution function for arithmetic coding.
        
        The LLM outputs logits (unnormalized log probabilities) for each token.
        These are converted to a CDF that the arithmetic coder can use.
        """
        logprobs = self.model.logits_to_logprobs(logits)
        probs = np.exp(logprobs).astype(np.float64)
        # Scale probabilities to integer frequencies for arithmetic coding
        freqs = np.maximum(1, np.round(FREQ_SCALE_FACTOR * probs))
        cum_freqs = np.cumsum(freqs)
        return cum_freqs

    def compress(self, uncompressed: bytes, window_overlap=0) -> bytes:
        """
        Compress data using the LLM's predictions.
        
        Process:
        1. Convert bytes to UTF-8 (using PUA encoding for invalid bytes)
        2. Tokenize the text using the model's tokenizer
        3. For each token, get the model's probability predictions
        4. Use arithmetic coding to compress based on these probabilities
        5. Handle context window limitations with sliding window approach
        """
        def sigint_handler(*_):
            """Handle Ctrl+C gracefully during compression."""
            nonlocal interrupted
            interrupted = True

        def process_logits(_, logits):
            """
            Process model logits for each token during generation.
            
            This callback is called for each token position. It:
            1. Gets the probability distribution from the model
            2. Encodes the actual next token using arithmetic coding
            3. Forces the model to predict that token (by setting its logit to infinity)
            """
            nonlocal next_token_idx
            
            # Check if we've finished processing all tokens
            if next_token_idx >= len(tokens):
                # Force EOS to stop generation
                logits[self.model.token_eos()] = np.inf
                return logits
                
            if interrupted and next_token_idx < len(tokens) - 1:
                next_token_idx = len(tokens) - 1
                if self.verbose:
                    print(file=sys.stderr)
                    
            next_token = tokens[next_token_idx]
            next_token_idx += 1
            
            # Get probability distribution and encode the token
            cdf = self.compute_cdf(logits)
            token_encoder.encode_symbol(cdf, next_token)
            progress_bar.update()
            
            # Force model to "predict" the correct token
            logits[next_token] = np.inf
            return logits

        def should_stop(tokens_so_far, logits):
            """Stop generation when we reach EOS or context limit or when we've processed all tokens."""
            return (
                next_token_idx >= len(tokens) or
                np.argmax(logits) == self.model.token_eos() or
                len(tokens_so_far) == self.model.n_ctx()
            )

        # Prepare for compression
        self.model.reset()
        tokens = self.model.tokenize(bytes_to_utf8(uncompressed), add_bos=False)
        tokens.append(self.model.token_eos())  # Add end-of-sequence token
        next_token_idx = 0
        token_encoder = Encoder()

        # Set up interrupt handling and progress tracking
        interrupted = False
        s = signal.signal(signal.SIGINT, sigint_handler)

        progress_bar = tqdm(
            total=len(tokens),
            mininterval=1 / 30,
            desc="Compressing",
            unit="tok",
            leave=False,
            dynamic_ncols=True,
            disable=not self.verbose,
        )

        # Process tokens using sliding window approach
        while next_token_idx < len(tokens):
            # Use overlapping context to maintain coherence across windows
            start_idx = max(0, next_token_idx - window_overlap)
            consume(
                self.model.generate(
                    tokens=[self.model.token_bos()] + tokens[start_idx:next_token_idx],
                    temp=0.0,  # Deterministic generation
                    logits_processor=[process_logits],
                    stopping_criteria=should_stop,
                )
            )
        progress_bar.close()

        # Finalize compression
        token_encoder.finish()
        compressed = token_encoder.get_encoded()

        signal.signal(signal.SIGINT, s)

        return compressed

    def tokenizer_adds_space_prefix(self):
        """
        Check if the tokenizer adds a space prefix to tokens.
        
        Some tokenizers (like LLaMA's) add a leading space to tokens.
        This affects how we handle the first token during decompression.
        """
        space = b" "
        double_space = b"  "
        tokenized = self.model.tokenize(space, add_bos=False)
        return self.model.detokenize(tokenized) == double_space

    def decompress(self, compressed: bytes, window_overlap=0) -> bytes:
        """
        Decompress data by reversing the compression process.
        
        Process:
        1. Initialize arithmetic decoder with compressed data
        2. For each position, get model's probability predictions
        3. Use decoder to determine which token was encoded
        4. Add token to context and continue until EOS
        5. Convert tokens back to bytes and handle UTF-8 decoding
        """
        def process_logits(_, logits):
            """
            Process logits during decompression.
            
            Unlike compression, here we don't know the next token.
            We use the probability distribution to decode it from
            the compressed stream.
            """
            nonlocal done
            
            if done:
                # Already finished, force EOS to stop generation
                logits[self.model.token_eos()] = np.inf
                return logits
            
            cdf = self.compute_cdf(logits)
            next_token = token_decoder.decode_symbol(cdf)
            logits[next_token] = np.inf
            
            if next_token == self.model.token_eos():
                done = True
                return logits
            
            # Convert token back to bytes
            next_utf8 = self.model.detokenize([next_token])
            
            # Handle tokenizer space prefix on first token
            if (
                len(seen_tokens) == 0
                and next_utf8.startswith(b" ")
                and self.tokenizer_adds_space_prefix()
            ):
                next_utf8 = next_utf8[1:]
            
            seen_tokens.append(next_token)
            next_bytes = utf8_to_bytes(utf8_decoder.decode(next_utf8))
            decompressed.extend(next_bytes)
            
            # Stream output in real-time if verbose
            if self.verbose:
                sys.stdout.buffer.write(next_bytes)
                sys.stdout.buffer.flush()
            return logits

        def should_stop(tokens_so_far, logits):
            """Stop when we hit EOS or context limit."""
            nonlocal done
            if np.argmax(logits) == self.model.token_eos():
                done = True
            return done or len(tokens_so_far) == self.model.n_ctx()

        # Initialize decompression
        self.model.reset()
        seen_tokens = []
        decompressed = bytearray()
        token_decoder = Decoder(compressed)
        utf8_decoder = codecs.getincrementaldecoder("utf-8")()
        done = False
        
        # Process compressed data with sliding window
        while not done:
            start_idx = max(0, len(seen_tokens) - window_overlap)
            consume(
                self.model.generate(
                    tokens=[self.model.token_bos()] + seen_tokens[start_idx:],
                    temp=0.0,
                    logits_processor=[process_logits],
                    stopping_criteria=should_stop,
                )
            )
        return decompressed


def make_arg_parser():
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM-powered lossless compression tool"
    )
    parser.add_argument("model_path", help="path to model file")
    parser.add_argument(
        "-f",
        "--compressed-format",
        choices=["binary", "base64"],
        help="format of compressed data (default: binary, except for interactive mode, which only supports base64)",
    )
    parser.add_argument(
        "-w",
        "--window-overlap",
        dest="overlap",
        default="0%",
        help="how much model context (as number of tokens or percentage of model context length) to maintain after filling the window. higher values increase compression ratio but decrease speed. must use same value for compression and decompression (default: 0%%)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=0,
        help="model context length (default: 0, which uses maximum supported by the model)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="number of model layers to offload to GPU (default: -1, which offloads all layers)",
    )
    parser.add_argument(
        "--use-mlock",
        default=False,
        action="store_true",
        help="use mlock to keep model in RAM (disabled by default)",
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-c",
        "--compress",
        dest="string",
        nargs="*",
        help="compress argument string (or stdin if no argument is provided)",
    )
    mode_group.add_argument(
        "-d",
        "--decompress",
        dest="compressed",
        nargs="*",
        help="decompress argument string (or stdin if no argument is provided)",
    )
    mode_group.add_argument(
        "-i",
        "--interactive",
        dest="interactive",
        default=False,
        action="store_true",
        help="show a prompt for interactive compression and decompression",
    )
    return parser


def robust_b64decode(input_bytes):
    """
    Robust base64 decoder that handles malformed input.
    
    Filters out invalid characters and adds padding as needed
    to handle user input that may not be perfectly formatted.
    """
    filtered_base64 = bytes(byte for byte in input_bytes if byte in BASE64)
    padded_base64 = filtered_base64 + b"A" * (-len(filtered_base64) % 4)
    return base64.b64decode(padded_base64)


def main():
    """
    Main entry point for the compression tool.
    
    Handles argument parsing, model initialization, and orchestrates
    the compression/decompression process based on user input.
    """
    parser = make_arg_parser()
    args = parser.parse_args()

    # Set default output format based on mode
    if args.compressed_format is None:
        args.compressed_format = "base64" if args.interactive else "binary"
    elif args.interactive and args.compressed_format != "base64":
        parser.error("interactive mode only supports base64 compressed data")

    # Initialize the compressor with the specified model
    compressor = LlamaZip(
        model_path=args.model_path,
        use_mlock=args.use_mlock,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=True,
    )

    # Parse window overlap parameter (can be tokens or percentage)
    try:
        if args.overlap.endswith("%"):
            percent = float(args.overlap[:-1])
            if not (0 <= percent <= 100):
                parser.error("window overlap must be in the range [0%, 100%]")
            window_overlap = int(percent / 100 * (compressor.model.n_ctx() - 1))
        else:
            window_overlap = int(args.overlap)
            if window_overlap < 0:
                window_overlap += compressor.model.n_ctx()
            if not (0 <= window_overlap < compressor.model.n_ctx()):
                parser.error(
                    f"window overlap must be in the range [{-compressor.model.n_ctx()}, {compressor.model.n_ctx() - 1}]"
                )
    except ValueError:
        parser.error(
            "window overlap must be an integer (number of tokens) or a percentage (of the model's context length)"
        )

    try:
        # Handle compression mode
        if args.string is not None:
            uncompressed = (
                " ".join(args.string).encode("utf-8")
                if args.string
                else sys.stdin.buffer.read()
            )
            compressed = compressor.compress(uncompressed, window_overlap)
            if args.compressed_format == "base64":
                compressed = base64.b64encode(compressed)
            sys.stdout.buffer.write(compressed)
            sys.stdout.buffer.flush()
        # Handle decompression mode
        elif args.compressed is not None:
            compressed = (
                args.compressed[0].encode("utf-8")
                if args.compressed
                else sys.stdin.buffer.read()
            )
            if args.compressed_format == "base64":
                compressed = robust_b64decode(compressed)
            compressor.decompress(compressed, window_overlap)
        # Handle interactive mode
        elif args.interactive:
            while True:
                try:
                    input_bytes = input("≥≥≥ ").encode("utf-8")
                except UnicodeDecodeError:
                    print(
                        "error: interactive mode only supports UTF-8 input",
                        file=sys.stderr,
                    )
                    continue
                # Auto-detect if input is base64 (for decompression) or text (for compression)
                if input_bytes and all(byte in BASE64_EQ for byte in input_bytes):
                    try:
                        compressed = robust_b64decode(input_bytes)
                        compressor.decompress(compressed, window_overlap)
                    except KeyboardInterrupt:
                        pass
                else:
                    compressed = compressor.compress(input_bytes, window_overlap)
                    compressed = base64.b64encode(compressed)
                    sys.stdout.buffer.write(compressed)
                    sys.stdout.buffer.flush()
                print("\n", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()