import torch
import json
import faiss
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ================= CONFIGURATION DE L'AGENT =================

# Chemins des modèles entraînés
MODEL_DIR = "./results_pretrain/final_pretrained_model"  # Doit être un causal LM
LORA_ADAPTER_DIR = "./results_finetune_lora/final_lora_adapters"

# Fichier de la base de données vectorielle pour la mémoire longue durée
VDB_FILE = "./data/lyra_memory.bin"

# Paramètres d'inférence
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8  # Plus haut = ton créatif

# ================= ARCHITECTURE DE L'AGENT =================

class LyraCognitiveAgent:
    def __init__(self):
        # 1. TOKENIZER
        print("1. Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        # 2. BASE MODEL (causal LM)
        print("2. Loading Base Model (FP16)...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype=torch.float16,
            device_map="auto"
        )

        # Ajustement vocab si LoRA a ajouté un token
        lora_vocab_size = 32001
        if len(self.tokenizer) < lora_vocab_size:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|dummy|>"]})
        self.base_model.resize_token_embeddings(lora_vocab_size)

        # 3. CHARGEMENT DES ADAPTATEURS LoRA
        print("3. Loading LoRA Adapters...")
        self.model = PeftModel.from_pretrained(self.base_model, LORA_ADAPTER_DIR)
        self.model = self.model.merge_and_unload()  # Fusion LoRA

        # 4. EMBEDDING MODEL (pour mémoire)
        print("4. Loading Embedding Model...")
        self.embedding_model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # 5. FAISS MEMORY
        self.vdb = self._load_vdb()
        self.memory_history = []

    # ==================== FAISS MEMORY ====================

    def _load_vdb(self):
        dim = self.embedding_model.get_sentence_embedding_dimension()
        if os.path.exists(VDB_FILE):
            print(f"5. Loading FAISS index from {VDB_FILE}...")
            return faiss.read_index(VDB_FILE)
        else:
            print("5. Creating new FAISS index...")
            return faiss.IndexFlatL2(dim)

    def _save_vdb(self):
        faiss.write_index(self.vdb, VDB_FILE)

    # ==================== TOOL FUNCTIONS ====================

    def web_search(self, query):
        print(f"[TOOL] Searching web for: '{query}'")
        return f"The search for '{query}' returned some placeholder info."

    def captcha_solver(self, image_path):
        print(f"[TOOL] Attempting to solve CAPTCHA at path: {image_path}")
        return {"success": True, "result": "Gryphon72"}

    TOOL_FUNCTIONS = {
        "web_search": web_search,
        "captcha_solver": captcha_solver
    }

    # ==================== MEMORY LOGIC ====================

    def _retrieve_context(self, prompt, user_id="user_A", k=3):
        if self.vdb.ntotal == 0:
            return ""
        query_vector = self.embedding_model.encode([prompt], convert_to_tensor=False)
        D, I = self.vdb.search(query_vector.astype('float32'), k)
        texts = [self.memory_history[idx]["text"] for idx in I[0] if idx < len(self.memory_history)]
        return "\n".join([f"[MEMORY]: {t}" for t in texts])

    def _store_memory(self, text, user_id="user_A", type="interaction"):
        if not text.strip():
            return
        embedding_vector = self.embedding_model.encode([text], convert_to_tensor=False)
        self.vdb.add(embedding_vector.astype('float32'))
        self.memory_history.append({
            "text": text,
            "user_id": user_id,
            "type": type,
            "timestamp": os.path.getmtime(VDB_FILE) if os.path.exists(VDB_FILE) else 0
        })

    # ==================== RESPONSE GENERATION ====================

    def generate_response(self, user_prompt, user_id="user_A"):
        context = self._retrieve_context(user_prompt, user_id)
        tool_schema = json.dumps(list(self.TOOL_FUNCTIONS.keys()))
        system_prompt = f"""
You are Lyra, a casual, personal companion.
[MEMORY CONTEXT]:
{context}
Available tools: {tool_schema}.
Use <|tool_call|>{{"tool": "function_name", "query": "..."}}<|endoftext|> for tools.
"""
        prompt = f"{system_prompt}\nUser: {user_prompt}{self.tokenizer.eos_token}Assistant:"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True
            )

        response_text = self.tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
        if response_text.startswith("<|tool_call|>"):
            return self._execute_tool_call(response_text, user_prompt, user_id)

        self._store_memory(f"User: {user_prompt}", user_id)
        self._store_memory(f"Assistant: {response_text}", user_id)
        self._internal_monologue(user_prompt, response_text, user_id)
        return response_text

    def _execute_tool_call(self, tool_call_text, user_prompt, user_id):
        try:
            json_str = tool_call_text.replace("<|tool_call|>", "").split(self.tokenizer.eos_token)[0].strip()
            call = json.loads(json_str)
            tool_name = call.get("tool")
            args = {k: v for k, v in call.items() if k != "tool"}
        except Exception as e:
            return f"Tool call parse error: {e}"

        if tool_name in self.TOOL_FUNCTIONS:
            tool_result = self.TOOL_FUNCTIONS[tool_name](self, **args)
            print(f"[TOOL EXECUTION] Result: {tool_result}")
            prompt_with_result = f"{user_prompt}\n[TOOL_OUTPUT]: {tool_result}\nAssistant:"
            input_ids = self.tokenizer.encode(prompt_with_result, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                final_output = self.model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True
                )
            final_resp = self.tokenizer.decode(final_output[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
            self._store_memory(f"Assistant (Tool Result): {final_resp}", user_id)
            return final_resp
        else:
            return f"Unknown tool '{tool_name}'"

    def _internal_monologue(self, user_prompt, assistant_response, user_id):
        prompt = f"""
Analyze last interaction (User: '{user_prompt}' | Assistant: '{assistant_response}').
Summarize key points to remember in 1-3 sentences starting with <|monologue|>.
"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        mono_text = self.tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=False).strip()
        if mono_text.startswith("<|monologue|>"):
            final_mono = mono_text.split(self.tokenizer.eos_token)[0].strip()
            self._store_memory(final_mono, user_id, "monologue")
            print(f"[COGNITION] Internal Monologue Stored: {final_mono}")

# ==================== MAIN LOOP ====================

if __name__ == "__main__":
    if not os.path.isdir(MODEL_DIR) or not os.path.isdir(LORA_ADAPTER_DIR):
        print("CRITICAL: Pre-trained model or LoRA adapters not found.")
    else:
        agent = LyraCognitiveAgent()
        print("\n--- Lyra Agent Ready (Type 'exit' or 'quit') ---")
        while True:
            user_input = input("You (Friend): ")
            if user_input.lower() in ["exit", "quit"]:
                agent._save_vdb()
                print("Lyra: Memory saved. Bye!")
                break
            resp = agent.generate_response(user_input)
            print(f"\nLyra: {resp}")
            print("-" * 50)
