# ReinforcementLearning

DDPG代码借鉴了morfan python的代码，在其基础上用PyTorch替换了TensoFlow

用能轻松解决gym倒立摆问题的DDPG模型在解决月球着陆器时难以收敛：
![62UTQ N_`(TCUZ6 FCF@}J6](https://user-images.githubusercontent.com/59995175/189302197-b850d8b2-2821-4b20-a1c4-76a7c8aa09d8.png)

DDPG月球着陆器解决：
1.修改Actor与Critic网络的结构：多增加一层FC和LN（BN的效果不好）
2.修改探索与利用的比例（噪声比例），因为月球着陆器相较于倒立摆要更复杂一点，收敛需要的回合数更多
3.将每一层网络的节点数扩大

![image](https://user-images.githubusercontent.com/59995175/189311811-ed9c1165-16f9-4089-9355-b51110133356.png)

最终在150 episodes左右模型收敛
