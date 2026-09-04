from __future__ import annotations
import json
import unicodedata
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
from functools import lru_cache

class ByteTokenizer:
    PAD=0
    BOS=1
    EOS=2
    UNK=3
    SYSTEM=4
    USER=5
    ASSISTANT=6
    END=7
    BYTE_OFFSET=8
    BYTE_VOCAB_SIZE=256
    SPECIAL_TOKENS={
        "<|pad|>":PAD,
        "<|bos|>":BOS,
        "<|eos|>":EOS,
        "<|unk|>":UNK,
        "<|system|>":SYSTEM,
        "<|user|>":USER,
        "<|assistant|>":ASSISTANT,
        "<|end|>":END,
    }
    ID_TO_SPECIAL={value:key for key,value in SPECIAL_TOKENS.items()}
    SPECIAL_TOKEN_LIST=list(SPECIAL_TOKENS.keys())
    VOCAB_SIZE=BYTE_OFFSET+BYTE_VOCAB_SIZE
    CJK_RANGES=[
        (0x4E00, 0x9FFF),
        (0x3400, 0x4DBF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x30000, 0x3134F),
        (0xF900, 0xFAFF),
        (0x3040, 0x309F),
        (0x30A0, 0x30FF),
        (0x31F0, 0x31FF),
    ]
    HIRAGANA_RANGE=(0x3040, 0x309F)
    KATAKANA_RANGE=(0x30A0, 0x30FF)
    PUNCTUATION_MAP={
        '。':'。',
        '、':'、',
        '「':'「',
        '」':'」',
        '『':'『',
        '』':'』',
        '（':'（',
        '）':'）',
        '【':'【',
        '】':'】',
        '・':'・',
        '…':'…',
        '！':'！',
        '？':'？',
        '～':'～',
    }

    def __init__(self, enable_cjk_aware: bool = True, enable_normalization: bool = True):
        self.vocab_size=self.VOCAB_SIZE
        self.pad_token_id=self.PAD
        self.bos_token_id=self.BOS
        self.eos_token_id=self.EOS
        self.unk_token_id=self.UNK
        self.system_token_id=self.SYSTEM
        self.user_token_id=self.USER
        self.assistant_token_id=self.ASSISTANT
        self.end_token_id=self.END
        self.enable_cjk_aware=enable_cjk_aware
        self.enable_normalization=enable_normalization
        self._cache_size=10000
        self._normalize_cache={}
        self._encode_cache={}
        self._decode_cache={}

    @property
    def pad_token(self)->str:
        return "<|pad|>"

    @property
    def bos_token(self)->str:
        return "<|bos|>"

    @property
    def eos_token(self)->str:
        return "<|eos|>"

    @property
    def unk_token(self)->str:
        return "<|unk|>"

    @property
    def system_token(self)->str:
        return "<|system|>"

    @property
    def user_token(self)->str:
        return "<|user|>"

    @property
    def assistant_token(self)->str:
        return "<|assistant|>"

    @property
    def end_token(self)->str:
        return "<|end|>"

    def is_special_token(self, token_id: int)->bool:
        try:
            token_id=int(token_id)
        except (TypeError, ValueError):
            return False
        return token_id in self.ID_TO_SPECIAL

    def is_byte_token(self, token_id: int)->bool:
        try:
            token_id=int(token_id)
        except (TypeError, ValueError):
            return False
        return self.BYTE_OFFSET<=token_id<self.BYTE_OFFSET+self.BYTE_VOCAB_SIZE

    def token_to_id(self, token: str)->int:
        if token in self.SPECIAL_TOKENS:
            return self.SPECIAL_TOKENS[token]
        return self.UNK

    def id_to_token(self, token_id: int)->str:
        try:
            token_id=int(token_id)
        except (TypeError, ValueError):
            return self.unk_token
        if token_id in self.ID_TO_SPECIAL:
            return self.ID_TO_SPECIAL[token_id]
        if self.is_byte_token(token_id):
            return f"<byte:{token_id-self.BYTE_OFFSET}>"
        return self.unk_token

    @lru_cache(maxsize=1024)
    def _is_cjk_character(self, code_point: int)->bool:
        for start, end in self.CJK_RANGES:
            if start<=code_point<=end:
                return True
        return False

    @lru_cache(maxsize=1024)
    def _is_hiragana(self, code_point: int)->bool:
        start, end=self.HIRAGANA_RANGE
        return start<=code_point<=end

    @lru_cache(maxsize=1024)
    def _is_katakana(self, code_point: int)->bool:
        start, end=self.KATAKANA_RANGE
        return start<=code_point<=end

    def _normalize_text(self, text: str)->str:
        if not self.enable_normalization:
            return text
        cache_key=text
        if cache_key in self._normalize_cache:
            if len(self._normalize_cache)>self._cache_size:
                self._normalize_cache.clear()
            return self._normalize_cache[cache_key]
        normalized=unicodedata.normalize('NFKC', text)
        normalized=normalized.replace('　', ' ')
        self._normalize_cache[cache_key]=normalized
        return normalized

    def _split_cjk_aware(self, text: str)->List[str]:
        if not self.enable_cjk_aware:
            return [text]
        parts=[]
        current=''
        prev_is_cjk=False
        for char in text:
            code_point=ord(char)
            is_cjk=self._is_cjk_character(code_point)
            is_hiragana=self._is_hiragana(code_point)
            is_katakana=self._is_katakana(code_point)
            is_japanese=(is_cjk or is_hiragana or is_katakana)
            is_whitespace=char.isspace()
            is_punctuation=char in self.PUNCTUATION_MAP
            if is_whitespace or is_punctuation:
                if current:
                    parts.append(current)
                    current=''
                if not is_whitespace:
                    parts.append(char)
                prev_is_cjk=False
            elif is_japanese:
                if current and not prev_is_cjk:
                    if not (self._is_hiragana(ord(current[-1])) or self._is_katakana(ord(current[-1]))):
                        parts.append(current)
                        current=char
                    else:
                        current+=char
                else:
                    current+=char
                prev_is_cjk=True
            else:
                if current and prev_is_cjk:
                    parts.append(current)
                    current=char
                    prev_is_cjk=False
                else:
                    current+=char
                    prev_is_cjk=False
        if current:
            parts.append(current)
        return [p for p in parts if p]

    def encode(self, text: str, add_bos: bool=False, add_eos: bool=True, add_special_tokens: bool=True)->List[int]:
        if not isinstance(text, str):
            text=str(text)
        cache_key=(text, add_bos, add_eos, add_special_tokens)
        if cache_key in self._encode_cache:
            if len(self._encode_cache)>self._cache_size:
                self._encode_cache.clear()
            return self._encode_cache[cache_key]
        text=self._normalize_text(text)
        tokens=[]
        if add_bos:
            tokens.append(self.BOS)
        if add_special_tokens:
            tokens.extend(self._encode_with_special_tokens(text))
        else:
            tokens.extend(self._encode_bytes(text))
        if add_eos:
            tokens.append(self.EOS)
        self._encode_cache[cache_key]=tokens
        return tokens

    def _encode_bytes(self, text: str)->List[int]:
        if not text:
            return []
        data=text.encode("utf-8", errors="replace")
        return [self.BYTE_OFFSET+byte for byte in data]

    def _encode_with_special_tokens(self, text: str)->List[int]:
        if not text:
            return []
        special_tokens=sorted(self.SPECIAL_TOKEN_LIST, key=len, reverse=True)
        tokens=[]
        i=0
        length=len(text)
        while i<length:
            matched=False
            for special_token in special_tokens:
                if text.startswith(special_token, i):
                    tokens.append(self.SPECIAL_TOKENS[special_token])
                    i+=len(special_token)
                    matched=True
                    break
            if matched:
                continue
            character=text[i]
            if self.enable_cjk_aware and (self._is_cjk_character(ord(character)) or self._is_hiragana(ord(character)) or self._is_katakana(ord(character))):
                tokens.extend(self._encode_bytes(character))
            else:
                if character.isspace():
                    tokens.extend(self._encode_bytes(character))
                elif character in self.PUNCTUATION_MAP:
                    tokens.extend(self._encode_bytes(character))
                else:
                    tokens.extend(self._encode_bytes(character))
            i+=1
        return tokens

    def decode(self, tokens: Iterable[int], skip_special_tokens: bool=True)->str:
        if tokens is None:
            return ""
        tokens_list=list(tokens)
        cache_key=(tuple(tokens_list), skip_special_tokens)
        if cache_key in self._decode_cache:
            if len(self._decode_cache)>self._cache_size:
                self._decode_cache.clear()
            return self._decode_cache[cache_key]
        result=[]
        raw=bytearray()

        def flush_bytes():
            if not raw:
                return
            result.append(self._decode_utf8_safely(bytes(raw)))
            raw.clear()

        for token in tokens_list:
            try:
                token=int(token)
            except (TypeError, ValueError):
                continue
            if token in self.ID_TO_SPECIAL:
                flush_bytes()
                if not skip_special_tokens:
                    result.append(self.ID_TO_SPECIAL[token])
                continue
            if self.is_byte_token(token):
                raw.append(token-self.BYTE_OFFSET)
                continue
            flush_bytes()
            if not skip_special_tokens:
                result.append(self.unk_token)

        flush_bytes()
        decoded=self._normalize_text("".join(result)) if self.enable_normalization else "".join(result)
        self._decode_cache[cache_key]=decoded
        return decoded

    @staticmethod
    def _decode_utf8_safely(data: bytes)->str:
        if not data:
            return ""
        result=[]
        i=0
        length=len(data)
        while i<length:
            first=data[i]
            if first<0x80:
                result.append(chr(first))
                i+=1
                continue
            if 0xC2<=first<=0xDF:
                needed=2
            elif 0xE0<=first<=0xEF:
                needed=3
            elif 0xF0<=first<=0xF4:
                needed=4
            else:
                i+=1
                continue
            if i+needed>length:
                break
            sequence=data[i:i+needed]
            if any(not 0x80<=sequence[j]<=0xBF for j in range(1, needed)):
                i+=1
                continue
            if needed==3:
                second=sequence[1]
                if first==0xE0 and second<0xA0:
                    i+=1
                    continue
                if first==0xED and second>=0xA0:
                    i+=1
                    continue
            elif needed==4:
                second=sequence[1]
                if first==0xF0 and second<0x90:
                    i+=1
                    continue
                if first==0xF4 and second>0x8F:
                    i+=1
                    continue
            try:
                result.append(sequence.decode("utf-8"))
            except UnicodeDecodeError:
                i+=1
                continue
            i+=needed
        return "".join(result)

    def encode_chat_message(self, role: str, content: str, add_eos: bool=False)->List[int]:
        role=str(role).lower().strip()
        if role=="system":
            role_id=self.SYSTEM
        elif role=="user":
            role_id=self.USER
        elif role=="assistant":
            role_id=self.ASSISTANT
        else:
            role_id=self.UNK
        tokens=[role_id]
        tokens.extend(self.encode(content, add_bos=False, add_eos=False, add_special_tokens=True))
        tokens.append(self.END)
        if add_eos:
            tokens.append(self.EOS)
        return tokens

    def encode_chat(self, messages, add_bos: bool=True, add_eos: bool=False)->List[int]:
        tokens=[]
        if add_bos:
            tokens.append(self.BOS)
        for message in messages:
            if not isinstance(message, dict):
                continue
            role=message.get("role", "user")
            content=message.get("content", "")
            if content is None:
                content=""
            tokens.extend(self.encode_chat_message(role, str(content), add_eos=False))
        if add_eos:
            tokens.append(self.EOS)
        return tokens

    def encode_assistant_prompt(self, messages, add_bos: bool=True)->List[int]:
        tokens=self.encode_chat(messages, add_bos=add_bos, add_eos=False)
        tokens.append(self.ASSISTANT)
        return tokens

    def create_assistant_mask(self, tokens: List[int])->List[int]:
        mask=[0]*len(tokens)
        in_assistant=False
        for i, token in enumerate(tokens):
            token=int(token)
            if token==self.ASSISTANT:
                in_assistant=True
                mask[i]=0
                continue
            if token==self.END:
                if in_assistant:
                    mask[i]=1
                in_assistant=False
                continue
            if token==self.EOS:
                if in_assistant:
                    mask[i]=1
                continue
            if in_assistant:
                mask[i]=1
        return mask

    def create_sft_labels(self, tokens: List[int], ignore_index: int=-100)->List[int]:
        if not tokens:
            return []
        mask=self.create_assistant_mask(tokens)
        return [int(token) if enabled else ignore_index for token, enabled in zip(tokens, mask)]

    def get_vocab(self):
        vocab=dict(self.SPECIAL_TOKENS)
        for byte in range(self.BYTE_VOCAB_SIZE):
            vocab[f"<byte:{byte}>"]=self.BYTE_OFFSET+byte
        return vocab

    def get_special_tokens(self):
        return dict(self.SPECIAL_TOKENS)

    def save(self, path):
        path=Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data={
            "type":"byte",
            "version":3,
            "vocab_size":self.vocab_size,
            "pad":self.PAD,
            "bos":self.BOS,
            "eos":self.EOS,
            "unk":self.UNK,
            "system":self.SYSTEM,
            "user":self.USER,
            "assistant":self.ASSISTANT,
            "end":self.END,
            "byte_offset":self.BYTE_OFFSET,
            "byte_vocab_size":self.BYTE_VOCAB_SIZE,
            "special_tokens":self.SPECIAL_TOKENS,
            "enable_cjk_aware":self.enable_cjk_aware,
            "enable_normalization":self.enable_normalization,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path=Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {path}")
        try:
            data=json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tokenizer file: {path}") from exc
        if data.get("type")!="byte":
            raise ValueError(f"Unsupported tokenizer type: {data.get('type')}")
        enable_cjk=data.get("enable_cjk_aware", True)
        enable_norm=data.get("enable_normalization", True)
        tokenizer=cls(enable_cjk_aware=enable_cjk, enable_normalization=enable_norm)
        if int(data.get("vocab_size", -1))!=tokenizer.vocab_size:
            raise ValueError(f"Tokenizer vocabulary mismatch: file={data.get('vocab_size')},runtime={tokenizer.vocab_size}")
        if int(data.get("byte_offset", -1))!=tokenizer.BYTE_OFFSET:
            raise ValueError(f"Tokenizer byte offset mismatch: file={data.get('byte_offset')},runtime={tokenizer.BYTE_OFFSET}")
        saved_special_tokens=data.get("special_tokens")
        if saved_special_tokens is not None:
            normalized_saved={str(key):int(value) for key, value in saved_special_tokens.items()}
            if normalized_saved!=tokenizer.SPECIAL_TOKENS:
                raise ValueError("Tokenizer special-token configuration mismatch.")
        return tokenizer

    def describe_tokens(self, tokens: Iterable[int]):
        result=[]
        for token in tokens:
            try:
                token=int(token)
            except (TypeError, ValueError):
                result.append({"id":None, "token":self.unk_token, "type":"invalid"})
                continue
            if token in self.ID_TO_SPECIAL:
                result.append({"id":token, "token":self.ID_TO_SPECIAL[token], "type":"special"})
            elif self.is_byte_token(token):
                result.append({"id":token, "token":f"<byte:{token-self.BYTE_OFFSET}>", "type":"byte"})
            else:
                result.append({"id":token, "token":self.unk_token, "type":"unknown"})
        return result

    def get_token_stats(self, tokens: Iterable[int])->Dict:
        tokens_list=list(tokens)
        special_count=sum(1 for t in tokens_list if self.is_special_token(t))
        byte_count=sum(1 for t in tokens_list if self.is_byte_token(t))
        unknown_count=sum(1 for t in tokens_list if not self.is_special_token(t) and not self.is_byte_token(t))
        return {
            "total":len(tokens_list),
            "special":special_count,
            "byte":byte_count,
            "unknown":unknown_count,
            "average_token_length":len(tokens_list)/max(1, len(self._decode_utf8_safely(b''.join(bytes([t-self.BYTE_OFFSET]) for t in tokens_list if self.is_byte_token(t))))) if tokens_list else 0,
        }

    def encode_batch(self, texts: List[str], add_bos: bool=False, add_eos: bool=True, pad_to_length: Optional[int]=None)->List[List[int]]:
        encoded=[self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]
        if pad_to_length is None:
            pad_to_length=max(len(seq) for seq in encoded) if encoded else 0
        return [seq+[self.PAD]*(pad_to_length-len(seq)) for seq in encoded]

    def decode_batch(self, token_batches: List[List[int]], skip_special_tokens: bool=True)->List[str]:
        return [self.decode(tokens, skip_special_tokens=skip_special_tokens) for tokens in token_batches]

def create_tokenizer(path="tokenizer/tokenizer.json"):
    tokenizer=ByteTokenizer(enable_cjk_aware=True, enable_normalization=True)
    tokenizer.save(path)
    print(f"Tokenizer created: {path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print("Special tokens:")
    for token, token_id in tokenizer.SPECIAL_TOKENS.items():
        print(f"  {token}: {token_id}")
    return tokenizer

def run_tests():
    tokenizer=ByteTokenizer(enable_cjk_aware=True, enable_normalization=True)
    print("="*60)
    print("Advanced NovaLLM Tokenizer Test")
    print("="*60)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()
    text="こんにちは、NovaLLMです。\nHello, world!"
    tokens=tokenizer.encode(text, add_bos=True, add_eos=True)
    restored=tokenizer.decode(tokens, skip_special_tokens=True)
    print("Original:")
    print(text)
    print()
    print("Tokens:")
    print(tokens)
    print()
    print("Decoded:")
    print(restored)
    print()
    assert restored==text, "Basic encode/decode test failed"
    chat_text="<|system|>\nあなたはNovaLLMです。<|end|>\n<|user|>\nこんにちは！<|end|>\n<|assistant|>\nこんにちは。<|end|>"
    chat_tokens=tokenizer.encode(chat_text, add_bos=True, add_eos=False)
    print("Chat tokens:")
    print(chat_tokens)
    print()
    print("Token description:")
    for item in tokenizer.describe_tokens(chat_tokens):
        print(item)
    print()
    chat_restored=tokenizer.decode(chat_tokens, skip_special_tokens=False)
    print("Chat decoded:")
    print(chat_restored)
    print()
    assert "<|system|>" in chat_restored
    assert "<|user|>" in chat_restored
    assert "<|assistant|>" in chat_restored
    assert "<|end|>" in chat_restored
    messages=[
        {"role":"system", "content":"あなたはNovaLLMです。"},
        {"role":"user", "content":"こんにちは"},
        {"role":"assistant", "content":"こんにちは！"},
    ]
    encoded_chat=tokenizer.encode_chat(messages, add_bos=True, add_eos=False)
    print("Encoded chat:")
    print(encoded_chat)
    print()
    assistant_prompt=tokenizer.encode_assistant_prompt(messages=[{"role":"system", "content":"あなたはNovaLLMです。"}, {"role":"user", "content":"こんにちは"}], add_bos=True)
    print("Assistant prompt:")
    print(assistant_prompt)
    print()
    decoded_prompt=tokenizer.decode(assistant_prompt, skip_special_tokens=False)
    print("Decoded assistant prompt:")
    print(decoded_prompt)
    print()
    assert decoded_prompt.endswith("<|assistant|>")
    sft_tokens=tokenizer.encode_chat([{"role":"user", "content":"1+1はいくつ？"}, {"role":"assistant", "content":"2です。"}], add_bos=True, add_eos=True)
    mask=tokenizer.create_assistant_mask(sft_tokens)
    labels=tokenizer.create_sft_labels(sft_tokens)
    print("SFT tokens:")
    print(sft_tokens)
    print()
    print("Assistant mask:")
    print(mask)
    print()
    print("SFT labels:")
    print(labels)
    print()
    assert len(mask)==len(sft_tokens)
    assert len(labels)==len(sft_tokens)
    assert 1 in mask
    batch_texts=["こんにちは", "Hello", "さようなら"]
    batch_encoded=tokenizer.encode_batch(batch_texts, add_bos=True, add_eos=True, pad_to_length=20)
    print("Batch encoding:")
    for text, encoded in zip(batch_texts, batch_encoded):
        print(f"  {text}: {encoded}")
    print()
    batch_decoded=tokenizer.decode_batch(batch_encoded, skip_special_tokens=True)
    print("Batch decoding:")
    for text, decoded in zip(batch_texts, batch_decoded):
        print(f"  {text} -> {decoded}")
    print()
    stats=tokenizer.get_token_stats(sft_tokens)
    print("Token statistics:")
    print(f"  Total tokens: {stats['total']}")
    print(f"  Special tokens: {stats['special']}")
    print(f"  Byte tokens: {stats['byte']}")
    print()
    japanese_text="日本語のテストです。ひらがなカタカナも対応しています。"
    jp_tokens=tokenizer.encode(japanese_text, add_bos=True, add_eos=True)
    jp_decoded=tokenizer.decode(jp_tokens, skip_special_tokens=True)
    print(f"Japanese test: {japanese_text}")
    print(f"Decoded: {jp_decoded}")
    assert jp_decoded==japanese_text
    print()
    assert tokenizer.PAD==0
    assert tokenizer.BOS==1
    assert tokenizer.EOS==2
    assert tokenizer.UNK==3
    assert tokenizer.SYSTEM==4
    assert tokenizer.USER==5
    assert tokenizer.ASSISTANT==6
    assert tokenizer.END==7
    assert tokenizer.BYTE_OFFSET==8
    assert tokenizer.vocab_size==264
    print("="*60)
    print("ALL ADVANCED TOKENIZER TESTS PASSED")
    print("="*60)

if __name__=="__main__":
    tokenizer=create_tokenizer()
    print()
    run_tests()
