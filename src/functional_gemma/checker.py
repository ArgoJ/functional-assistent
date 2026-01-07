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

        assistant_msg = item['messages'][2]
        if 'tool_calls' in assistant_msg and assistant_msg['tool_calls']:
            expected_tool = assistant_msg['tool_calls'][0]['function']['name']
            other_tools = [tool['function']['name'] for tool in TOOLS if tool['function']['name'] != expected_tool]
            
            if f"call:{expected_tool}" in output and \
                    not any(tool in output for tool in other_tools):
                print(f"  -> ✅ correct tool call: {expected_tool}")
                success_count += 1
            elif f"call:{expected_tool}" in output \
                    and any(tool in output for tool in other_tools):
                print(f"  -> ⚠️ tool is correct {expected_tool}, but other_tool exists in output")
                print(f"     Output: {output}")
            else:
                print(f"  -> ❌ wrong (expected tool '{expected_tool}' missing output)")
                print(f"     Output: {output}")
        else:
            if "<start_function_call>" not in output:
                print("  -> ✅ correct text response")
                success_count += 1
            else:
                print("  -> ❌ wrong (hallucinated a tool call when text was expected)")
                print(f"     Output: {output}")

    print(f"Success rate: {success_count} / {len(dataset)}")