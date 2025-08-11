# from transformers import CLIPTokenizer

# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32") # 使用しているGLIPのモデルに合ったトークナイザーを指定
# prompt_text = "あなたのプロンプトのテキストをここに入力します。"
# tokens = tokenizer.encode(prompt_text)
# num_tokens = len(tokens)
# print(f"トークン数: {num_tokens}")


# from transformers import AutoTokenizer

# # Hugging Faceで確認したGLIPモデルのIDを指定
# # 例: microsoft/glip-t もしくは microsoft/glipv2-swin-large-doc
# model_id = "GLIPModel/GLIP" # あなたが使用しているGLIPモデルの正確なIDに置き換えてください

# # AutoTokenizerが自動的に適切なトークナイザーをロード
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# print(f"ロードされたトークナイザーのクラス: {type(tokenizer)}")

# # トークン数をテストするプロンプト
# prompt_text = "A cat sitting on a couch with a remote control."
# tokens = tokenizer.encode(prompt_text, add_special_tokens=True) # 特殊トークンもカウントに含めるためTrue

# print(f"プロンプト: \"{prompt_text}\"")
# print(f"トークン数: {len(tokens)}")
# print(f"トークンID (一部): {tokens[:10]}...") # 全て表示すると長くなるので一部
# print(f"デコードされたトークン (視覚的な確認): {tokenizer.convert_ids_to_tokens(tokens)}")

# # 最大トークン長も確認しておくと良いでしょう
# print(f"トークナイザーの最大入力長 (model_max_length): {tokenizer.model_max_length}")

from transformers import CLIPTokenizer # CLIPのトークナイザーを直接インポート

# GLIPがベースとしているOpenAIのCLIPモデルのトークナイザーを指定
# GLIPのバージョンや論文によって、'base' か 'large' かが異なる場合があります。
# 一般的には 'openai/clip-vit-base-patch32' がよく使われます。
tokenizer_name = "openai/clip-vit-base-patch32"

try:
    # CLIPTokenizer を直接ロード
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)

    print(f"ロードされたトークナイザーのクラス: {type(tokenizer)}")

    # テスト用のプロンプト
    prompt_text = "Main Cable, Main Cable (Multiple Cables), Hanger Rope, Stays, Stay Support, Parallel Wire Strand, Strand Rope, Spiral Rope, Locked Coil Rope, PC Steel Wire, Semi-Parallel Wire Strand, Zinc-Plated Steel Wire, Cable Band, Saddle Cover, Cable Cover, Socket Cover, Anchor Cover, Tower Saddle, Spray Saddle, Intersection Fitting, Stay Grip Fitting, Socket (Open Type), Socket (Rod Anchor Type), Socket (Pressure Anchor Type), End Clamp (SG Socket), End Clamp (Screw Type), Crimp Anchor, Wire Clip Anchor, Embedded Anchor, Cable Anchor, Socket Anchor, Rod Anchor, Anchor Rod, Rod Thread Part, Rod Anchor Nut, Fixing Bolt, Anchor Piece, Shackle, Turnbuckle, Wire Clip, Connecting Fitting, Wire Seeging, Rubber Boot, Stainless Band, Grout Injection Port"
    
    # トークン化とトークン数の取得
    # add_special_tokens=True で特殊トークン（CLS, SEPなど）もカウントに含めます
    tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

    print(f"プロンプト: \"{prompt_text}\"")
    print(f"トークン数: {len(tokens)}")
    print(f"トークンID (一部): {tokens[:10]}...") # 全て表示すると長くなるので一部
    print(f"デコードされたトークン (視覚的な確認): {tokenizer.convert_ids_to_tokens(tokens)}")

    # トークナイザーの最大入力長も確認しておくと良いでしょう
    print(f"トークナイザーの最大入力長 (model_max_length): {tokenizer.model_max_length}")

except Exception as e:
    print(f"エラーが発生しました: {e}")
    print("指定されたトークナイザーのパスが間違っているか、ネットワークの問題です。")



from transformers import AutoTokenizer # または BertTokenizer

# GLIPが使用しているBERTモデルのIDを指定
tokenizer_name = "bert-base-uncased" 

try:
    # AutoTokenizer が bert-base-uncased に対応する BertTokenizer を自動的にロードします
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # または直接 BertTokenizer をインポートしてロード:
    # from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    print(f"ロードされたトークナイザーのクラス: {type(tokenizer)}")

    # テスト用のプロンプト
    # prompt_text = "Cable Component, Cable Support Structure, Cable Anchorage, Cable Rods and Bolts, Other Cable Accessories"
    prompt_text = "Main Cable, Main Cable (Multiple Cables), Hanger Rope, Stays, Stay Support, Parallel Wire Strand, Strand Rope, Spiral Rope, Locked Coil Rope, PC Steel Wire, Semi-Parallel Wire Strand, Zinc-Plated Steel Wire, Cable Band, Saddle Cover, Cable Cover, Socket Cover, Anchor Cover, Tower Saddle, Spray Saddle, Intersection Fitting, Stay Grip Fitting, Socket (Open Type), Socket (Rod Anchor Type), Socket (Pressure Anchor Type), End Clamp (SG Socket), End Clamp (Screw Type), Crimp Anchor, Wire Clip Anchor, Embedded Anchor, Cable Anchor, Socket Anchor, Rod Anchor, Anchor Rod, Rod Thread Part, Rod Anchor Nut, Fixing Bolt, Anchor Piece, Shackle, Turnbuckle, Wire Clip, Connecting Fitting, Wire Seeging, Rubber Boot, Stainless Band, Grout Injection Port"
    
    # トークン化とトークン数の取得
    # add_special_tokens=True で特殊トークン（CLS, SEPなど）もカウントに含めます
    tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

    print(f"プロンプト: \"{prompt_text}\"")
    print(f"トークン数: {len(tokens)}")
    print(f"トークンID (一部): {tokens[:10]}...") # 全て表示すると長くなるので一部
    print(f"デコードされたトークン (視覚的な確認): {tokenizer.convert_ids_to_tokens(tokens)}")

    # トークナイザーの最大入力長も確認しておくと良いでしょう
    # BERTは通常512トークンが最大ですが、GLIP側で256に制限している場合はそちらに従います
    print(f"トークナイザーの最大入力長 (model_max_length): {tokenizer.model_max_length}")

except Exception as e:
    print(f"エラーが発生しました: {e}")
    print("指定されたトークナイザーのパスが間違っているか、ネットワークの問題です。")