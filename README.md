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

   VeriOS-Agent-7B: [Hugging Face](https://huggingface.co/wuuuuuz/VeriOS-Agent-7B)

   VeriOS-Agent-32B: [Hugging Face](https://huggingface.co/wuuuuuz/VeriOS-Agent-32B)

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
If you find our work useful, please cite our paper:
```bibtex
@article{wu2025verios,
  title={VeriOS: Query-Driven Proactive Human-Agent-GUI Interaction for Trustworthy OS Agents},
  author={Wu, Zheng and Li, Ziyang and Zhang, Yuhang and Zhang, Haowei and Liu, Jiaqi and Li, Yixuan and Zhao, Pu and Wang, Yujia and Xu, Yifan and Zhao, Yuyang and others},
  journal={arXiv preprint arXiv:2509.07553},
  year={2025}
}
```
