/- WitCert 形式化根模块。四层结构见 README.md。 -/
import WitCert.Refinement      -- L4:proof–kernel refinement(优先机器检查)
import WitCert.SoftmaxTV       -- L1:softmax 似然比与 sound e-form 界
import WitCert.UniformDither   -- L2:均匀抖动 sub-Gaussian proxy
import WitCert.RequestBudget   -- L3:请求级 union budget
import WitCert.Contract     -- L5:带度量类型的契约演算(论文2 中心定理)
import WitCert.Bridges      -- L6:打分桥 + softmax 桥(tv_le_eform 实例化)+ 仿射松弛
import WitCert.Ledger       -- L7:请求级风险账本(望远镜权重 + 未知长度 soundness + 覆盖率置信)
import WitCert.Ville        -- L8:有限 Ω Ville + e-process(E3;零测度论)
import WitCert.Radius       -- L9:E2 半径分析核心 —— 两点 Hoeffding 引理(常数 1/8)+ Jensen(F3)
import WitCert.McDiarmid    -- L10:有限 Ω McDiarmid —— E2 尾项的概率核心(F3 Stage 2)
import WitCert.SharedRead   -- L11:共享读风险语义 —— δ 按写事件计,读者数不放大(H3)
