# federated‑learning‑course‑design
基于树莓派4B的横向联邦学习实验平台，FedAvg算法，TCP自定义通信协议

## 项目简介
搭建树莓派4B物理硬件集群，实现真实设备下横向联邦学习流程。树莓派客户端完成本地训练，通过自定义TCP‑Socket上传模型参数；服务端执行FedAvg加权聚合并下发全局模型，完成IID/NIID场景对比实验。
> 硬件部署、通信框架、核心代码均为本人独立完成；

## 本人工作
- 完成树莓派集群部署、静态IP、SSH、ARM环境适配与局域网组网调试
- 使用原生Socket实现TCP自定义通信，处理并发接入、报文封装、异常断连
- 实现客户端本地训练、服务端FedAvg聚合，完成IID/NIID数据集划分
- 开展多组对照实验，解决边缘算力受限、参数传输开销等工程问题

## 技术栈
Python3、PyTorch(ARM)、Socket(TCP)、Raspberry Pi OS、NumPy

## 运行方式
1. 修改`config.py`配置IP、端口及训练超参
2. 启动服务端：`python server.py`
3. 树莓派客户端：`python client.py`，自动执行联邦训练
