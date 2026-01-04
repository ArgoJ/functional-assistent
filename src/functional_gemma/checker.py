def check_success_rate(dataset, model, tokenizer, TOOLS):
    success_count = 0
    for idx, item in enumerate(dataset):
        messages = [
            item["messages"][0],
            item["messages"][1],
        ]

        inputs = tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, return_dict=True, return_tensors="pt")

        out = model.generate(**inputs.to(model.device), pad_token_id=tokenizer.eos_token_id, max_new_tokens=128)
        output = tokenizer.decode(out[0][len(inputs["input_ids"][0]) :], skip_special_tokens=False)

        print(f"{idx+1} Prompt: {item['messages'][1]['content']}")
        print(f"  Output: {output}")

        expected_tool = item['messages'][2]['tool_calls'][0]['function']['name']
        other_tool = "search_knowledge_base" if expected_tool == "search_google" else "search_google"

        if expected_tool in output and other_tool not in output:
            print("  `-> ✅ correct!")
            success_count += 1
        elif expected_tool not in output:
            print(f"  -> ❌ wrong (expected '{expected_tool}' missing)")
        else:
            if output.startswith(f"<start_function_call>call:{expected_tool}"):
                print(f"  -> ⚠️ tool is correct {expected_tool}, but other_tool exists in output")
            else:
                print(f"  -> ❌ wrong (hallucinated '{other_tool}')")

    print(f"Success rate: {success_count} / {len(dataset)}")