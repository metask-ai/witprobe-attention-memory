/- 发布正门:两个快照的四条接缝不变量,`by decide` 编译失败 = 不许出包。
   数据由 tools/release_export.py 机械导出(ReleaseData.lean);
   挂载点:make_release.py 出包关(--check 与落盘前各一次)。
   不得 import 进 WitCert 根 —— 工程层,不计入论文定理数。-/
import WitCert.ReleaseData

namespace WitCert.Release

theorem p1_release_publishable : publishable snapshot_p1 = true := by decide

theorem p2_release_publishable : publishable snapshot_p2 = true := by decide

end WitCert.Release
