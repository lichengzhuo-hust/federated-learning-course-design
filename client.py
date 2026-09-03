# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import socket
import pickle
import struct
import torch
import time
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from common import MNIST_CNN

TOTAL_ROUNDS = 10

def get_data_loader(client_id):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=False
    )
    total_size = len(full_dataset)
    chunk_size = total_size // 3
    start_idx = client_id * chunk_size
    end_idx = start_idx + chunk_size
    client_subset = Subset(full_dataset, range(start_idx, end_idx))
    return DataLoader(client_subset, batch_size=64, shuffle=True, num_workers=0)

def local_train(client_id, global_weights, train_loader):
    model = MNIST_CNN()
    model.load_state_dict(global_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    batch_idx = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"  [Client {client_id}] Batch {batch_idx}, Loss = {loss.item():.4f}")
        batch_idx += 1
    state_dict = model.state_dict()
    del model
    return state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, default="10.121.188.1")
    args = parser.parse_args()
    port = 65432
    print(f"[Main] 已启动，连接至 {args.server}:{port}")

    for round_idx in range(TOTAL_ROUNDS):
        print(f"\n=== 第 {round_idx + 1} / {TOTAL_ROUNDS} 轮 ===")
        for client_id in range(3):
            print(f"[Main] 扮演 Client {client_id} ...")
            train_loader = get_data_loader(client_id)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(None)
                sock.connect((args.server, port))
                sock.sendall(f"CLIENT_{client_id}".encode())
                print(f"[Client {client_id}] 已连上，等待模型下发...")
                
                raw_len = sock.recv(4)
                if not raw_len: continue
                msg_len = struct.unpack('!I', raw_len)[0]
                data = b''
                while len(data) < msg_len:
                    packet = sock.recv(msg_len - len(data))
                    if not packet: break
                    data += packet
                
                global_weights = pickle.loads(data)
                print(f"[Client {client_id}] 收到模型，开始训练！")
                trained_state_dict = local_train(client_id, global_weights, train_loader)
                
                send_data = pickle.dumps(trained_state_dict)
                sock.sendall(struct.pack('!I', len(send_data)) + send_data)
                sock.close()
                time.sleep(1)
            except Exception as e:
                print(f"[Client {client_id}] 异常: {e}")
                continue
    print("[Main] 所有轮次执行完毕。")
