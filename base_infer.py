import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 基本パラメータ
model_name = "cyberagent/open-calm-7b"
dataset = "kunishou/databricks-dolly-15k-ja"


# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 評価モード
model.eval()

# プロンプトテンプレートの準備
def generate_prompt(data_point):
    if data_point["input"]:
        result = f"""### 指示:
{data_point["instruction"]}

### 入力:
{data_point["input"]}

### 回答:
"""
    else:
        result = f"""### 指示:
{data_point["instruction"]}

### 回答:
"""

    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result

# テキスト生成関数の定義
def generate(instruction,input=None,maxTokens=256):
    # 推論
    prompt = generate_prompt({'instruction':instruction,'input':input})
    input_ids = tokenizer(prompt, 
        return_tensors="pt", 
        truncation=True, 
        add_special_tokens=False).input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids, 
        max_new_tokens=maxTokens, 
        do_sample=True,
        temperature=0.1, 
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
    )
    outputs = outputs[0].tolist()
    # print(tokenizer.decode(outputs).replace('<NL>','\n'))

    # EOSトークンにヒットしたらデコード完了
    if tokenizer.eos_token_id in outputs:
        eos_index = outputs.index(tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[:eos_index])

        # レスポンス内容のみ抽出
        sentinel = "### 回答:"
        sentinelLoc = decoded.find(sentinel)
        if sentinelLoc >= 0:
            result = decoded[sentinelLoc+len(sentinel):]
            print(result.replace("<NL>", "\n"))  # <NL>→改行
        else:
            print('Warning: Expected prompt template to be emitted.  Ignoring output.')
    else:
        print('Warning: no <eos> detected ignoring output')
        
for i in range(10):
    generate("自然言語処理とは？")