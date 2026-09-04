from __future__ import annotations
import argparse
from pathlib import Path
import torch
from config import MODEL,RUNTIME
from model import NovaLM
from tokenizer import ByteTokenizer

class NovaInference:
    def __init__(self,checkpoint:str,device:str|None=None):
        self.device=device or RUNTIME.device
        self.tokenizer=ByteTokenizer()
        if self.tokenizer.vocab_size!=264:
            raise ValueError(f"Expected tokenizer vocabulary size 264,got {self.tokenizer.vocab_size}")
        MODEL.vocab_size=self.tokenizer.vocab_size
        self.model=NovaLM(MODEL)
        checkpoint_path=Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_data=torch.load(checkpoint_path,map_location=self.device,weights_only=False)
        if isinstance(checkpoint_data,dict):
            checkpoint_vocab=checkpoint_data.get("tokenizer_vocab_size")
            if checkpoint_vocab is not None and int(checkpoint_vocab)!=self.tokenizer.vocab_size:
                raise ValueError(f"Tokenizer vocabulary mismatch: checkpoint={checkpoint_vocab},current={self.tokenizer.vocab_size}")
            state=checkpoint_data.get("model",checkpoint_data)
        else:
            state=checkpoint_data
        self.model.load_state_dict(state,strict=True)
        self.model.to(self.device)
        self.model.eval()
        print(f"[MODEL] Loaded: {checkpoint_path}")

    def build_prompt(self,messages):
        if not isinstance(messages,list):
            messages=list(messages)
        normalized=[]
        for message in messages:
            if not isinstance(message,dict):
                continue
            role=str(message.get("role","user")).strip().lower()
            content=str(message.get("content",""))
            if role not in ("system","user","assistant"):
                role="user"
            normalized.append({"role":role,"content":content})
        tokens=self.tokenizer.encode_chat(normalized,add_bos=True,add_eos=False)
        tokens.append(self.tokenizer.ASSISTANT)
        return tokens

    def _clean_generated_text(self,text:str):
        if not text:
            return ""
        for marker in ("<|end|>","<|user|>","<|assistant|>","<|system|>"):
            if marker in text:
                text=text.split(marker,1)[0]
        text=text.replace("\x00","")
        text=text.replace("\ufffd","")
        return text.strip()

    @torch.no_grad()
    def generate(self,messages,max_new_tokens=256,temperature=0.7,top_k=50,top_p=0.95):
        tokens=self.build_prompt(messages)
        if not tokens:
            tokens=[self.tokenizer.BOS,self.tokenizer.ASSISTANT]
        if len(tokens)>=MODEL.max_seq_len:
            tokens=tokens[-MODEL.max_seq_len:]
            if tokens[0]!=self.tokenizer.BOS:
                tokens[0]=self.tokenizer.BOS
        input_ids=torch.tensor([tokens],dtype=torch.long,device=self.device)
        output=self.model.generate(input_ids,max_new_tokens=max_new_tokens,temperature=temperature,top_k=top_k,top_p=top_p,eos_token_id=self.tokenizer.END)
        generated=output[0,input_ids.shape[1]:].tolist()
        text=self.tokenizer.decode(generated,skip_special_tokens=True)
        return self._clean_generated_text(text)

def interactive(engine:NovaInference):
    messages=[]
    print()
    print("="*70)
    print("NovaLLM interactive mode")
    print("終了: exit / quit")
    print("="*70)
    while True:
        try:
            user=input("\nYou > ")
        except (EOFError,KeyboardInterrupt):
            print()
            break
        if user.strip().lower() in ("exit","quit"):
            break
        if not user.strip():
            continue
        messages.append({"role":"user","content":user})
        try:
            answer=engine.generate(messages)
        except Exception as e:
            print(f"\nNovaLLM error: {e}")
            messages.pop()
            continue
        print(f"\nNova > {answer}")
        messages.append({"role":"assistant","content":answer})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",default="checkpoints/sft_final.pt")
    parser.add_argument("--device",default=None)
    parser.add_argument("--max-new-tokens",type=int,default=256)
    args=parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError("Checkpoint not found: "+args.checkpoint)
    engine=NovaInference(checkpoint=args.checkpoint,device=args.device)
    interactive(engine)

if __name__=="__main__":
    main()
