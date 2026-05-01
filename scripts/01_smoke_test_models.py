import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "gptneo_1p3b": "EleutherAI/gpt-neo-1.3B",
    "rwkv4": "RWKV/rwkv-4-169m-pile",
}

PROMPT = "Hello, this is a smoke test."


def summarize_state(state):
    if state is None:
        print("state: None")
        return

    print("state type:", type(state))

    if isinstance(state, (list, tuple)):
        print("state length:", len(state))
        preview_items = state[:5]
        for i, item in enumerate(preview_items):
            if hasattr(item, "shape"):
                print(f"state[{i}].shape:", tuple(item.shape))
            else:
                print(f"state[{i}] type:", type(item))
    else:
        if hasattr(state, "shape"):
            print("state.shape:", tuple(state.shape))
        else:
            print("state value type:", type(state))


def move_inputs_to_model_device(inputs, model):
    try:
        device = model.device
        return {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        return inputs


def test_model(model_name: str, model_id: str) -> None:
    print("=" * 100)
    print(f"Testing model: {model_name}")
    print(f"Model ID: {model_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Tokenizer loaded.")

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Tokenizer pad_token set to eos_token.")

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        print("Model loaded.")

        inputs = tokenizer(PROMPT, return_tensors="pt")
        inputs = move_inputs_to_model_device(inputs, model)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )

        print("Forward pass succeeded.")
        print("logits.shape:", tuple(outputs.logits.shape))

        if getattr(outputs, "hidden_states", None) is not None:
            print("num_hidden_states:", len(outputs.hidden_states))
            print("last_hidden_state.shape:", tuple(outputs.hidden_states[-1].shape))
        else:
            print("hidden_states: None")

        if hasattr(outputs, "state"):
            summarize_state(outputs.state)
        else:
            print("state attribute: not present")

        try:
            gen_inputs = tokenizer(PROMPT, return_tensors="pt")
            gen_inputs = move_inputs_to_model_device(gen_inputs, model)

            generated = model.generate(
                **gen_inputs,
                max_new_tokens=20,
                do_sample=False,
            )
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            print("generation sample:", decoded)
        except Exception as gen_e:
            print(f"Generation test skipped/failed: {gen_e}")

    except Exception as e:
        print(f"[FAILED] {model_name}: {e}")
        traceback.print_exc()


def main():
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        try:
            print("cuda device name:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    for name, model_id in MODELS.items():
        test_model(name, model_id)


if __name__ == "__main__":
    main()