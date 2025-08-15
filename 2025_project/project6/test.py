from phe import paillier
import random
import hashlib

# ------------------ 参与方P2：生成Paillier密钥对 ------------------
paillier_pub_key, paillier_priv_key = paillier.generate_paillier_keypair(n_length=512)

# ------------------ 参与方P1：准备本地数据集 ------------------
local_identifiers = ["password1", "123456", "qwerty"]
secret_k1 = random.randint(2, 1000)  # P1的秘密随机数
mod_prime = 2 ** 521 - 1  # 用于模运算的大素数


def identifier_hash(input_str):
    """将字符串标识符转换为哈希整数"""
    return int(hashlib.sha256(input_str.encode()).hexdigest(), 16)


# P1对本地标识符进行盲化处理
p1_blinded = [pow(identifier_hash(v), secret_k1, mod_prime) for v in local_identifiers]
random.shuffle(p1_blinded)  # 打乱顺序增强隐私

# ------------------ 参与方P2：处理本地数据并返回 ------------------
remote_identifiers = ["123456", "letmein", "password2"]
risk_scores = [5, 10, 7]  # 对应每个标识符的风险分数
secret_k2 = random.randint(2, 1000)  # P2的秘密随机数

# P2对本地标识符进行盲化
p2_blinded = [pow(identifier_hash(w), secret_k2, mod_prime) for w in remote_identifiers]
# 使用Paillier公钥加密风险分数
encrypted_scores = [paillier_pub_key.encrypt(score) for score in risk_scores]

# 对P1发来的盲化数据进行二次盲化
double_blind_set = [pow(v_blind, secret_k2, mod_prime) for v_blind in p1_blinded]
# 打包待发送给P1的数据并打乱顺序
p2_data = list(zip(p2_blinded, encrypted_scores))
random.shuffle(p2_data)
random.shuffle(double_blind_set)

# ------------------ 参与方P1：计算交集并累加风险值 ------------------
matching_encrypted = []
for w_blind, enc_score in p2_data:
    # 对P2的盲化标识符进行二次盲化
    w_double = pow(w_blind, secret_k1, mod_prime)
    # 检查是否属于交集
    if w_double in double_blind_set:
        matching_encrypted.append(enc_score)

# 利用Paillier加法同态性累加加密的风险值
if matching_encrypted:
    sum_encrypted = matching_encrypted[0]
    for enc in matching_encrypted[1:]:
        sum_encrypted += enc
else:
    sum_encrypted = None

# ------------------ 参与方P2：解密得到总风险值 ------------------
if sum_encrypted is not None:
    total_risk_score = paillier_priv_key.decrypt(sum_encrypted)
    print(f"交集标识符的总风险值: {total_risk_score}")
else:
    print("未发现交集标识符，总风险值为0")
