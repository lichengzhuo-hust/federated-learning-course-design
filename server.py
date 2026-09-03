# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

import socket
import pickle
import struct
import time
from collections import OrderedDict
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from common import MNIST_CNN

HOST = '0.0.0.0'
PORT = 65432
TOTAL_CLIENTS = 3
NUM_ROUNDS = 10

def send_obj(conn, obj):
    data = pickle.dumps(obj)
    conn.sendall(struct.pack('!I', len(data)) + data)

def recv_obj(conn):
    try:
        raw_len = conn.recv(4)
        if not raw_len: return None
        msg_len = struct.unpack('!I', raw_len)[0]
        data = b''
        while len(data) < msg_len:
            packet = conn.recv(msg_len - len(data))
            if not packet: return None
            data += packet
        return pickle.loads(data)
    except Exception:
        return None

def aggregate(models):
    if not models: return None
    avg_dict = OrderedDict()
    keys = models[0].keys() 
    for key in keys:
        avg_dict[key] = torch.stack([m[key].float() for m in models]).mean(dim=0)
    return avg_dict

def evaluate_accuracy(model_state_dict):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=False
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    model = MNIST_CNN()
    model.load_state_dict(model_state_dict)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    accuracy = 100. * correct / total
    print(f"[Server] 🔍 全局模型在测试集上的准确率: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"[Server] 启动成功，将执行 {NUM_ROUNDS} 轮...\n")

    global_model = MNIST_CNN().state_dict()
    accuracy_history = []

    for round_idx in range(NUM_ROUNDS):
        print(f"\n================= 第 {round_idx + 1} / {NUM_ROUNDS} 轮 =================")
        client_models = []

        for cid in range(TOTAL_CLIENTS):
            print(f"[Server] 等待 Client {cid} 前来连接...")
            conn, addr = server_socket.accept()
            try:
                client_id_bytes = conn.recv(1024).decode().strip()
                received_id = int(client_id_bytes.split('_')[1])
                if received_id != cid:
                    conn.close()
                    continue

                print(f"[Server] Client {cid} 已连接，下发模型...")
                send_obj(conn, global_model)

                print(f"[Server] 等待 Client {cid} 训练完成...")
                trained_state_dict = recv_obj(conn)
                
                if trained_state_dict is not None:
                    client_models.append(trained_state_dict)
                    print(f"[Server] ✅ Client {cid} 返回成功！")
                else:
                    print(f"[Server] ❌ Client {cid} 异常返回")
                conn.close()
            except Exception as e:
                print(f"[Server] Client {cid} 异常: {e}")
                if conn: conn.close()

        if len(client_models) == TOTAL_CLIENTS:
            print(f"\n[Server] 🧮 聚合成功！")
            global_model = aggregate(client_models)
            acc = evaluate_accuracy(global_model)
            accuracy_history.append(acc)
        else:
            print(f"\n[Server] ⚠️ 聚合跳过。")

    print("\n=== 全部 10 轮训练完成！===")

    if len(accuracy_history) > 0:
        print("[Server] 📊 正在生成收敛曲线图...")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, marker='o', linestyle='-', color='b')
        plt.title('Federated Learning Convergence Curve (MNIST)')
        plt.xlabel('Communication Round')
        plt.ylabel('Test Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('convergence_curve.png')
        print(f"[Server] ✅ 收敛曲线已成功保存为: convergence_curve.png")

        # 【保存最终的模型权重文件】
        torch.save(global_model, 'mnist_fed_model.pth')
        print(f"[Server] ✅ 训练好的模型权重已成功保存为: mnist_fed_model.pth")

    server_socket.close()
