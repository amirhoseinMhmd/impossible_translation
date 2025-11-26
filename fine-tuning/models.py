from transformers import (
    GPT2LMHeadModel, GPT2Config,
    GPTNeoXForCausalLM, GPTNeoXConfig,
    LlamaForCausalLM, LlamaConfig,
    MistralForCausalLM, MistralConfig,
    Phi3ForCausalLM, Phi3Config,
    AutoTokenizer,
)


def create_model_from_scratch(model_type='pythia-160m'):
    model_configs = {

        'pythia-160m': {
            'class': GPTNeoXForCausalLM,
            'config_class': GPTNeoXConfig,
            'tokenizer': 'EleutherAI/pythia-160m',
            'config': {
                'hidden_size': 768,
                'num_hidden_layers': 12,
                'num_attention_heads': 12,
                'intermediate_size': 3072,
                'hidden_act': 'gelu',
                'rotary_pct': 0.25,
                'rotary_emb_base': 10000,
                'max_position_embeddings': 2048,
                'initializer_range': 0.02,
                'layer_norm_eps': 1e-5,
                'use_cache': True,
                'tie_word_embeddings': False,
            }
        },

        'olmo-0.5b': {
            'class': LlamaForCausalLM,
            'config_class': LlamaConfig,
            'tokenizer': 'allenai/OLMo-1B-hf',
            'config': {
                'hidden_size': 896,
                'num_hidden_layers': 16,
                'num_attention_heads': 14,
                'num_key_value_heads': 14,
                'intermediate_size': 4864,
                'hidden_act': 'silu',
                'max_position_embeddings': 2048,
                'initializer_range': 0.02,
                'rms_norm_eps': 1e-5,
                'use_cache': True,
                'tie_word_embeddings': False,
                'rope_theta': 10000.0,
            }
        },

        'tinyllama-1.1b': {
            'class': LlamaForCausalLM,
            'config_class': LlamaConfig,
            'tokenizer': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'config': {
                'hidden_size': 2048,
                'num_hidden_layers': 22,
                'num_attention_heads': 32,
                'num_key_value_heads': 4,
                'intermediate_size': 5632,
                'hidden_act': 'silu',
                'max_position_embeddings': 2048,
                'initializer_range': 0.02,
                'rms_norm_eps': 1e-5,
                'use_cache': True,
                'tie_word_embeddings': False,
                'rope_theta': 10000.0,
            }
        },

        'llama2-7b': {
            'class': LlamaForCausalLM,
            'config_class': LlamaConfig,
            'tokenizer': 'meta-llama/Llama-2-7b-hf',
            'config': {
                'hidden_size': 4096,
                'num_hidden_layers': 32,
                'num_attention_heads': 32,
                'num_key_value_heads': 32,
                'intermediate_size': 11008,
                'hidden_act': 'silu',
                'max_position_embeddings': 4096,
                'initializer_range': 0.02,
                'rms_norm_eps': 1e-6,
                'use_cache': True,
                'tie_word_embeddings': False,
                'rope_theta': 10000.0,
            }
        },

        'llama2-13b': {
            'class': LlamaForCausalLM,
            'config_class': LlamaConfig,
            'tokenizer': 'meta-llama/Llama-2-13b-hf',
            'config': {
                'hidden_size': 5120,
                'num_hidden_layers': 40,
                'num_attention_heads': 40,
                'num_key_value_heads': 40,  # MHA
                'intermediate_size': 13824,
                'hidden_act': 'silu',
                'max_position_embeddings': 4096,
                'initializer_range': 0.02,
                'rms_norm_eps': 1e-6,
                'use_cache': True,
                'tie_word_embeddings': False,
                'rope_theta': 10000.0,
            }
        },

        'llama3-8b': {
            'class': LlamaForCausalLM,
            'config_class': LlamaConfig,
            'tokenizer': 'meta-llama/Meta-Llama-3-8B',
            'config': {
                'hidden_size': 4096,
                'num_hidden_layers': 32,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,  # GQA
                'intermediate_size': 14336,
                'hidden_act': 'silu',
                'max_position_embeddings': 8192,
                'initializer_range': 0.02,
                'rms_norm_eps': 1e-5,
                'use_cache': True,
                'tie_word_embeddings': False,
                'rope_theta': 500000.0,
            }
        },

        'llama3.1-8b': {
            'class': LlamaForCausalLM,
            'config_class': LlamaConfig,
            'tokenizer': 'meta-llama/Meta-Llama-3.1-8B',
            'config': {
                'hidden_size': 4096,
                'num_hidden_layers': 32,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,
                'intermediate_size': 14336,
                'hidden_act': 'silu',
                'max_position_embeddings': 131072,
                'initializer_range': 0.02,
                'rms_norm_eps': 1e-5,
                'use_cache': True,
                'tie_word_embeddings': False,
                'rope_theta': 500000.0,
            }
        },

        'mistral-7b': {
            'class': MistralForCausalLM,
            'config_class': MistralConfig,
            'tokenizer': 'mistralai/Mistral-7B-v0.1',
            'config': {
                'hidden_size': 4096,
                'num_hidden_layers': 32,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,
                'intermediate_size': 14336,
                'hidden_act': 'silu',
                'max_position_embeddings': 32768,
                'initializer_range': 0.02,
                'rms_norm_eps': 1e-5,
                'use_cache': True,
                'tie_word_embeddings': False,
                'rope_theta': 10000.0,
                'sliding_window': 4096,
            }
        },

        'phi3-mini': {
            'class': Phi3ForCausalLM,
            'config_class': Phi3Config,
            'tokenizer': 'microsoft/Phi-3-mini-4k-instruct',
            'config': {
                'hidden_size': 3072,
                'num_hidden_layers': 32,
                'num_attention_heads': 32,
                'num_key_value_heads': 32,
                'intermediate_size': 8192,
                'hidden_act': 'silu',
                'max_position_embeddings': 4096,
                'initializer_range': 0.02,
                'rms_norm_eps': 1e-5,
                'use_cache': True,
                'tie_word_embeddings': False,
                'rope_theta': 10000.0,
                'attention_dropout': 0.0,
                'embd_pdrop': 0.0,
                'resid_pdrop': 0.0,
            }
        },

        'phi4': {
            'class': Phi3ForCausalLM,
            'config_class': Phi3Config,
            'tokenizer': 'microsoft/phi-4',
            'config': {
                'hidden_size': 5120,
                'num_hidden_layers': 40,
                'num_attention_heads': 40,
                'num_key_value_heads': 10,
                'intermediate_size': 17920,
                'hidden_act': 'silu',
                'max_position_embeddings': 16384,
                'initializer_range': 0.02,
                'rms_norm_eps': 1e-5,
                'use_cache': True,
                'tie_word_embeddings': False,
                'rope_theta': 250000.0,
                'attention_dropout': 0.0,
                'embd_pdrop': 0.0,
                'resid_pdrop': 0.0,
            }
        },

        'phi4-mini': {
            'class': Phi3ForCausalLM,
            'config_class': Phi3Config,
            'tokenizer': 'microsoft/Phi-4-mini-instruct',
            'config': {
                'hidden_size': 3072,
                'num_hidden_layers': 32,
                'num_attention_heads': 24,
                'num_key_value_heads': 8,
                'intermediate_size': 8192,
                'hidden_act': 'silu',
                'max_position_embeddings': 4096,
                'initializer_range': 0.02,
                'rms_norm_eps': 1e-5,
                'use_cache': True,
                'tie_word_embeddings': True,
                'rope_theta': 10000.0,
                'attention_dropout': 0.0,
                'embd_pdrop': 0.0,
                'resid_pdrop': 0.0,
            }
        },

        'gpt2-small': {
            'class': GPT2LMHeadModel,
            'config_class': GPT2Config,
            'tokenizer': 'gpt2',
            'config': {
                'n_embd': 256,
                'n_layer': 6,
                'n_head': 4,
                'n_positions': 256,
                'activation_function': 'gelu_new',
                'resid_pdrop': 0.1,
                'embd_pdrop': 0.1,
                'attn_pdrop': 0.1,
                'layer_norm_epsilon': 1e-5,
                'initializer_range': 0.02,
            }
        },
        'gpt2-base': {
            'class': GPT2LMHeadModel,
            'config_class': GPT2Config,
            'tokenizer': 'gpt2',
            'config': {
                'n_embd': 768,
                'n_layer': 12,
                'n_head': 12,
                'n_positions': 1024,
                'activation_function': 'gelu_new',
                'resid_pdrop': 0.1,
                'embd_pdrop': 0.1,
                'attn_pdrop': 0.1,
                'layer_norm_epsilon': 1e-5,
                'initializer_range': 0.02,
            }
        },
    }

    if model_type not in model_configs:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_configs.keys())}")

    config_info = model_configs[model_type]

    # Load tokenizer
    print(f"Loading tokenizer from {config_info['tokenizer']}...")
    tokenizer = AutoTokenizer.from_pretrained(config_info['tokenizer'], trust_remote_code=True)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model config
    model_config_params = config_info['config'].copy()
    model_config_params['vocab_size'] = tokenizer.vocab_size
    model_config_params['bos_token_id'] = tokenizer.bos_token_id
    model_config_params['eos_token_id'] = tokenizer.eos_token_id
    model_config_params['pad_token_id'] = tokenizer.pad_token_id

    # Create config object
    config_class = config_info['config_class']
    model_config = config_class(**model_config_params)

    # Create model with random weights
    model_class = config_info['class']
    model = model_class(model_config)

    # Print model info
    num_params = model.num_parameters()
    print(f"\n{'=' * 60}")
    print(f"Created {model_type} from scratch")
    print(f"{'=' * 60}")
    print(f"Parameters: {num_params:,}")
    print(f"Model class: {model_class.__name__}")
    print(f"Hidden size: {model_config_params.get('hidden_size', model_config_params.get('n_embd', 'N/A'))}")
    print(f"Layers: {model_config_params.get('num_hidden_layers', model_config_params.get('n_layer', 'N/A'))}")
    print(
        f"Attention heads: {model_config_params.get('num_attention_heads', model_config_params.get('n_head', 'N/A'))}")
    print(f"{'=' * 60}\n")

    return model, tokenizer


AVAILABLE_MODELS = [
    'pythia-160m',
    'olmo-0.5b',
    'tinyllama-1.1b',
    'llama2-7b',
    'llama2-13b',
    'llama3-8b',
    'llama3.1-8b',
    'mistral-7b',
    'phi3-mini',
    'phi4',
    'phi4-mini',
    'gpt2-small',
    'gpt2-base',
]

if __name__ == "__main__":
    test_models = ['pythia-160m', 'gpt2-small']

    for model_type in test_models:
        print(f"\nTesting {model_type}...")
        model, tokenizer = create_model_from_scratch(model_type)
        print(f"Success! Model has {model.num_parameters():,} parameters")

    print("\n\nAvailable models:")
    for m in AVAILABLE_MODELS:
        print(f"  - {m}")