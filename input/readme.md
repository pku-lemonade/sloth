### adj_mat.csv

mesh 网络拓扑的邻接矩阵

网络大小为 4 × 4， 邻接矩阵大小 16 × 16

##### 对应 DVC 算法中的 A

### path.csv

每一行表示一条端到端通信

前 24 列代表通信选择的路径， 最后一列是平均通信延迟（cycle数）

该文件中的路径由 id 表示

##### 对应 DVC 算法中的 SR 和 EED

### map.csv

记录了从实际链路 (start_core_id, end_core_id) 到链路编号 link_id 的映射

双向边只记录一次，默认 start_core_id 小于 end_core_id