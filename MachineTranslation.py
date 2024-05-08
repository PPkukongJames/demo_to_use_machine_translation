class Machine_Translation :
    _instance = None
    model = None
    tokenizer = None
    engine = None

    def __new__(cls, name):
        if cls._instance is None:
            cls._instance = super(Machine_Translation, cls).__new__(cls)
            if name == 'mbart_fb':
                from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
                cls.model = MBartForConditionalGeneration.from_pretrained("google/madlad400-10b-mt")
                cls.tokenizer = MBart50TokenizerFast.from_pretrained("google/madlad400-10b-mt")
                cls.engine = cls._instance.mbart_fb
            elif name == 'pythai':
                from pythainlp.translate import Translate
                cls.engine = Translate('th','en').translate
        return cls._instance

    def mbart_fb(self, article_en):
        self.tokenizer.src_lang = "th_TH"
        encoded_hi = self.tokenizer(article_en, return_tensors="pt", padding=True, truncation=True)
        generated_tokens = self.model.generate(
            **encoded_hi,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"]
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)