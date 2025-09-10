# VeriOS
Research code for the paper "VeriOS: Query-Driven Proactive Human-Agent-GUI Interaction for Trustworthy OS Agents".

Paper link: [https://arxiv.org/abs/2509.07553](https://arxiv.org/abs/2509.07553)

## 🚀 Quick Start
### 1. Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Wuzheng02/VeriOS
   ```
2. Navigate into the project directory:
   ```bash
   cd VeriOS
   ```
3. Download the VeriOS-Bench dataset:
  
   [Google Drive]()

4. Download the pre-trained models:

   VeriOS-Agent-7B: [https://huggingface.co/wuuuuuz/VeriOS-Agent-7B](https://huggingface.co/wuuuuuz/VeriOS-Agent-7B)

   VeriOS-Agent-32B: [https://huggingface.co/wuuuuuz/VeriOS-Agent-32B](https://huggingface.co/wuuuuuz/VeriOS-Agent-32B)

### 2. Evaluation
1. Evaluate VeriOS-Agent performance:
   ```bash
   python test_interaction_loop.py --model_path /path/to/VeriOS-Agent --json_path /path/to/test.json
   ```
2. Evaluate dual-agent system performance:
   ```bash
   python dual_agent.py --model_path1 /path/to/scenarioagent --model_path2 /path/to/actionagent --json_path /path/to/test.json
   ```
3. Evaluate other baselines:
   ```bash
   python test_loop_{name}.py --model_path /path/to/agent --json_path /path/to/test.json
   ```

### 3. Training
This work is based on full fine-tuning of LLMs using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We gratefully acknowledge the support from the LLaMA-Factory project.

To reproduce the training process of VeriOS-Agent from scratch:
1. Replace the `.yaml` files in the LLaMA-Factory repository with those provided in this repository.
2. Follow the official training tutorials provided in the [LLaMA-Factory repository](https://github.com/hiyouga/LLaMA-Factory).

## 📋 Citation
```bibtex
@article{wu2025verios,
  title={VeriOS: Query-Driven Proactive Human-Agent-GUI Interaction for Trustworthy OS Agents},
  author={Zheng Wu and Heyuan Huang and Xingyu Lou and Xiangmou Qu and Pengzhou Cheng and Zongru Wu and Weiwen Liu and Weinan Zhang and Jun Wang and Zhaoxiang Wang and Zhuosheng Zhang},
  journal={arXiv preprint arXiv:2509.07553},
  year={2025}
}
```
