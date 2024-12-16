from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# {"id":1,"title":"雇用工讀生","question":"公司辦展覽欲雇用工讀生看顧器材，分為兩個時段各四個小時\n\r\nQ1, 若工讀生自願連續工作八個小時不休息，這樣是否違反勞基法？ 那若雙方協議可以嗎？ 如果不行，休息時間是否需要照算薪水？（例如來上班八個小時，安排休息一個小時，薪水是算八小時還是七小時）\n\r\nQ2, 工讀生若在國定假日上班，薪水也是否給雙倍？\n\r\nQ3, 如果要跟工讀生訂定工作契約來確保工讀生須對設備保管責任，哪類相關契約可以參考？"
query = "雇用工讀生，公司辦展覽欲雇用工讀生看顧器材，分為兩個時段各四個小時\n\r\nQ1, 若工讀生自願連續工作八個小時不休息，這樣是否違反勞基法？ 那若雙方協議可以嗎？ 如果不行，休息時間是否需要照算薪水？（例如來上班八個小時，安排休息一個小時，薪水是算八小時還是七小時）\n\r\nQ2, 工讀生若在國定假日上班，薪水也是否給雙倍？\n\r\nQ3, 如果要跟工讀生訂定工作契約來確保工讀生須對設備保管責任，哪類相關契約可以參考？"
prompt = f"請將以下查詢轉化為更適合查詢搜索法律條文的形式:\n{query}\n\n優化後的查詢:"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
optimized_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(optimized_query)