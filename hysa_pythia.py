#!/usr/bin/env python3
"""
HySA (Hybrid Semantic Architecture) — Validation on EleutherAI/pythia-70m
Author: Francesco Mariani
Includes exact examples from Handwritten Blueprints (Sheet 1 & Sheet 2):
  - Polysemantic Intelligence (Np): DNS, IP, White Pages, Phone Directory
  - Semantic Hypersphere Radial Vectors (A, B, C, D): CAT, GATTO, KG, MACCHINA
  - Monosemantic Memory Inodes (Nk / Nm): '8.8.8.8', '24', 'JOHN', and IANA Ports
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Hardware Backend: {device.upper()}")
print("[*] Caricamento EleutherAI/pythia-70m in memoria...")

model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

TARGET_LAYER = 3  # Layer 3 del residual stream (dim = 512)

# ==============================================================================
# 1. VERIFICA DELL'ISOMORFISMO POLISEMANTICO (Foglio 1: DNS / IP / Pagine Bianche)
# ==============================================================================
print("\n" + "="*72)
print("[+] FASE 1: Isomorfismo Polisemantico in Np (DNS, IP, White Pages, Phone)")
print("="*72)

ISOMORPHIC_CONCEPTS = {
    "DNS": "A directory system that resolves a symbolic domain name to a target address is",
    "IP_Routing": "A network directory table that maps a symbolic host label to a target address is",
    "White_Pages": "A printed directory book that resolves a person's name to a target address is",
    "Phone_Directory": "A telecom lookup index that maps a subscriber name to a target number is"
}

poly_vectors = {}
with torch.no_grad():
    for name, prompt in ISOMORPHIC_CONCEPTS.items():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        vec = outputs.hidden_states[TARGET_LAYER][0, -1, :].clone()
        poly_vectors[name] = vec / vec.norm()

print("[*] Matrice di Similarita' Coseno nello Spazio Latente (Layer 3):")
c_names = list(poly_vectors.keys())
print(f"{'':16}", "".join([f"{n:>15}" for n in c_names]))
for n1 in c_names:
    row = f"{n1:<16}" + "".join([f"{torch.dot(poly_vectors[n1], poly_vectors[n2]).item():15.4f}" for n2 in c_names])
    print(row)

# ==============================================================================
# 2. RAGGI VETTORIALI DELL'IPERSFERA SEMANTICA (Foglio 1: CAT, GATTO, KG, MACCHINA)
# ==============================================================================
print("\n" + "="*72)
print("[+] FASE 1-B: Raggi Vettoriali (A, B, C, D) nell'Ipersfera Semantica")
print("="*72)

HYPERSPHERE_RAYS = {
    "Center (CAT)": "In domestic animals, the small feline pet is called a cat",
    "Ray_A (GATTO)": "In Italian language, the small feline domestic pet is called a gatto",
    "Ray_B (KG)": "The physical body mass of the domestic animal is measured in kg",
    "Ray_C (MACCHINA)": "Parked outside on the street, the mechanical vehicle is a macchina"
}

ray_vecs = {}
with torch.no_grad():
    for label, text in HYPERSPHERE_RAYS.items():
        inp = tokenizer(text, return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True)
        v = out.hidden_states[TARGET_LAYER][0, -1, :].clone()
        ray_vecs[label] = v / v.norm()

center_v = ray_vecs["Center (CAT)"]
for label, v in ray_vecs.items():
    dist = torch.norm(center_v - v).item()
    cos_sim = torch.dot(center_v, v).item()
    print(f" -> {label:<18} | Distanza Sferica dal Centro: d = {dist:.4f} | CosSim = {cos_sim:.4f}")

# ==============================================================================
# 3. INDICIZZAZIONE MONOSEMANTICA (Foglio 1 & 2: 8.8.8.8, 24, JOHN + Porte IANA)
# ==============================================================================
print("\n" + "="*72)
print("[+] FASE 2: Indicizzazione Centroidi Monosemantici Nk/Nm sull'Ipersfera")
print("="*72)

MONOSEMANTIC_INODES = [
    {
        "inode_id": "DNS_PORT",
        "context": "The standard network protocol used to translate domain names to IP addresses is DNS, and its default port number is",
        "literal_value": " 53"
    },
    {
        "inode_id": "SSH_PORT",
        "context": "The secure network protocol used for remote command-line login and administration is SSH, and its default port number is",
        "literal_value": " 22"
    },
    {
        "inode_id": "HTTP_PORT",
        "context": "The foundation data communication protocol for the World Wide Web is HTTP, and its default plaintext port number is",
        "literal_value": " 80"
    },
    {
        "inode_id": "PUBLIC_DNS_IP",
        "context": "The primary Google public DNS resolver IPv4 literal address is",
        "literal_value": " 8.8.8.8"
    },
    {
        "inode_id": "SUBNET_MASK_BITS",
        "context": "A standard Class C local network prefix of 255.255.255.0 has a CIDR bit length of",
        "literal_value": " 24"
    },
    {
        "inode_id": "ADMIN_PROPER_NAME",
        "context": "In the system registry, the primary root user first name literal is",
        "literal_value": " JOHN"
    }
]

nm_centroids = {}
nm_literals = {}

with torch.no_grad():
    for item in MONOSEMANTIC_INODES:
        inputs = tokenizer(item["context"], return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        raw_vec = outputs.hidden_states[TARGET_LAYER][0, -1, :].clone()
        mu_m = raw_vec / raw_vec.norm()
        nm_centroids[item["inode_id"]] = mu_m
        nm_literals[item["inode_id"]] = item["literal_value"]
        print(f" -> Inode Nk [{item['inode_id']:<17}] | Norma: {mu_m.norm().item():.2f} | Valore Atomico Rigido: '{item['literal_value']}'")

inode_keys = list(nm_centroids.keys())
dists = [
    torch.norm(nm_centroids[k1] - nm_centroids[k2]).item()
    for i, k1 in enumerate(inode_keys) for j, k2 in enumerate(inode_keys) if i < j
]
d_min = min(dists)
T_t = 0.46  # Raggio della Temperatura Topologica

print(f"\n[*] Distanza minima inter-Inode sull'Ipersfera (d_min): {d_min:.4f}")
print(f"[*] Raggio Temperatura Topologica configurato (T_t): {T_t:.2f}")

# ==============================================================================
# 4. ESECUZIONE QUERY/API DETERMINISTICA SULL'IPERSFERA SEMANTICA
# ==============================================================================
print("\n" + "="*72)
print("[+] FASE 3: Verifica tramite Query/API nell'Ipersfera Semantica B(mu_k, T_t)")
print("="*72)

QUERIES = [
    ("For remote server management and terminal administration, the SSH default port number is", "SSH_PORT"),
    ("The plaintext hyper-text transport protocol HTTP default port number is", "HTTP_PORT"),
    ("In network configuration, the primary Google public DNS resolver IPv4 literal address is", "PUBLIC_DNS_IP"),
    ("A standard local subnet mask of 255.255.255.0 has a CIDR bit length of", "SUBNET_MASK_BITS"),
    ("In the database system registry, the primary root user first name literal is", "ADMIN_PROPER_NAME"),
    ("An unverified telemetry protocol for subterranean speleological cave sensors default port number is", "UNINDEXED")
]

for prompt, expected_inode in QUERIES:
    print(f"\n[Query] \"{prompt}\"")
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        raw_q = outputs.hidden_states[TARGET_LAYER][0, -1, :].clone()
        q_interrogated = raw_q / raw_q.norm()
        raw_token_id = outputs.logits[0, -1, :].argmax().item()
        raw_pred = tokenizer.decode(raw_token_id)

    best_dist = float("inf")
    nearest_inode = None
    for k, mu_m in nm_centroids.items():
        d = torch.norm(q_interrogated - mu_m).item()
        if d < best_dist:
            best_dist = d
            nearest_inode = k

    print(f"  |-- Predizione Grezza Np (Senza Verifica): '{raw_pred}'")
    print(f"  |-- Punto Interrogato -> Inode Nk piu' vicino: [{nearest_inode}] a distanza d = {best_dist:.4f}")

    if best_dist <= T_t:
        confirmed_val = nm_literals[nearest_inode]
        print(f"  \\-- [API HYSA: CONFERMATO] d ({best_dist:.4f}) <= T_t ({T_t:.2f})")
        print(f"      Consistenza deterministica applicata -> Output certificato: '{confirmed_val}'")
    else:
        print(f"  \\-- [API HYSA: SEMANTIC FAULT] d ({best_dist:.4f}) > T_t ({T_t:.2f})")
        print(f"      Punto interrogato fuori dall'Ipersfera Semantica. Allucinazione bloccata.")