# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "llama-cpp-python==0.3.16",
#   "termcolor==3.1.0",
# ]
# ///

"""
Complete compression breakdown with Phi3
"""

# Standard imports
import math

# Library imports
from llama_cpp import Llama
from termcolor import colored


def compression_breakdown(text):
    """Breaks down compression into multiple steps"""
    
    print("🔍 LLM-based Compression Analysis")
    print("=" * 50)
    print(f"📝 Input: \"{text}\"")
    print(f"📏 Size: {len(text)} characters, {len(text.encode('utf-8'))} bytes")
    
    # Load model for tokenization
    print("\n🤖 Loading Phi-3 model...")
    model = Llama(
        model_path="Phi-3.1-mini-128k-instruct-Q4_K_M.gguf",
        n_ctx=100,
        verbose=False,
        logits_all=True
    )
    
    # Get actual tokenization
    tokens = model.tokenize(text.encode('utf-8'), add_bos=True)
    token_strings = [model.detokenize([token]).decode('utf-8', errors='ignore') for token in tokens]
    
    print(f"\n🪙 Tokenization:")
    print(f"   Token IDs: {tokens}")
    print(f"   Tokens: {token_strings}")
    print(f"   Count: {len(tokens)} tokens")
    
    # Calculate real probabilities from the model
    print(f"\n🧠 Calculating probabilities...")
    probs = []
    context = ""
    
    for i in range(len(tokens)):
        token_str = token_strings[i]
        try:
            response = model.create_completion(
                prompt=[model.token_bos()] + tokens[:i],
                max_tokens=1,
                temperature=0.0,
                logprobs=100,
                echo=False
            )
            
            # Extract logprobs from response
            if ('choices' in response and response['choices'] and 
                'logprobs' in response['choices'][0] and 
                response['choices'][0]['logprobs'] and
                'top_logprobs' in response['choices'][0]['logprobs'] and
                response['choices'][0]['logprobs']['top_logprobs']):
                
                top_logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
                
                # Find our target token's probability
                target_prob = None
                for token_text, logprob in top_logprobs.items():
                    if token_text.strip() == token_str.strip():
                        target_prob = math.exp(logprob)
                        break
                
                if target_prob is None:
                    # Token not in top logprobs, estimate low probability
                    target_prob = 0.001
                    print(f"   Token {i+1} {colored(token_str.strip(), 'blue')} not in top predictions, estimated P={target_prob*100:.1f}%")
                else:
                    print(f"   Token {i+1} {colored(token_str.strip(), 'blue')} found in predictions, P={target_prob*100:.1f}%")

                probs.append(target_prob)
                
            else:
                # Fallback if no logprobs available
                target_prob = 0.01
                probs.append(target_prob)
                print(f"   Token {i+1} '{token_str}': No logprobs available, estimated P={target_prob:.3f}")
                
        except Exception as e:
            # Fallback probability if there's an error
            target_prob = 0.01
            probs.append(target_prob)
            raise e
            print(f"   Token {i+1} '{token_str}': Error getting probability ({type(e).__name__}: '{str(e)}'), estimated P={target_prob:.3f}")
        
        # Add this token to context for next prediction
        context += token_str
    
    print(f"\n📊 Probability Analysis:")
    total_bits = 0
    
    for i, (token_str, prob) in enumerate(zip(token_strings, probs)):
        bits = -math.log2(prob)
        total_bits += bits
        print(f"   {i+1}. {token_str:20s} P≈{prob*100:.1f}% \t→\t {bits:.2f} bits")

    # Compression metrics
    original_bits = len(text.encode('utf-8')) * 8
    
    print(f"\n📈 Compression Results:")
    print(f"   Original: {original_bits} bits ({len(text.encode('utf-8'))} bytes)")
    print(f"   Compressed: {total_bits:.2f} bits ({total_bits/8:.2f} bytes)")
    print(f"   Ratio: {original_bits/total_bits:.1f}:1")
    print(f"   Savings: {(1 - total_bits/original_bits)*100:.0f}%")

if __name__ == "__main__":
    compression_breakdown("There is no compression algorithm for experience")
