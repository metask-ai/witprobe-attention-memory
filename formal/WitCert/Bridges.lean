/-
  WitCert 形式化 · L6:**两座桥** —— 让 存储见证 → 注意力 TV 的组合合法化

  背景:度量类型检查(WitCert/Contract.lean)拒绝了论文2 原来的链 ——
  存储段输出 kv_entry:rel_witness(无量纲比值),选择段输入 attn_dist:tv(概率质量),
  `a₂·b₁+b₂` 把两个没有共同单位的数相加。要让请求级证书复活,必须补出:

    ① witness → score   打分桥:|Δ(scale·⟨q,k⟩)| ≤ scale·‖q‖·‖Δk‖(Cauchy–Schwarz)
       系数 scale·‖q‖ 是**运行时量**,由 dsv4-qnorm 探针采(p75)。
    ② score → tv        softmax 桥:‖Δs‖∞ ≤ ε ⟹ TV ≤ ½(e^{2ε}−1)
       **不另证** —— 它是论文1 主定理 tv_le_eform 的指数倾斜实例化:
       softmax(s') 恰是 softmax(s) 的 e^{−ε} 倾斜,均匀 c ≡ ε 时 A = e^ε。
    ③ 仿射松弛:½(e^{2ε}−1) ≤ e^{2ε₀}·ε 当 0 ≤ ε ≤ ε₀
       Chain 只会仿射组合,非线性界要以"带前提的仿射系数"进链;
       前提 ε ≤ ε₀ 由运行时检查(ε₀ 取实测上界),这正是 partial 档的定义。

  这三条对应 Python 侧 contracts.py 的 proof 字段;桥的**系数**是否如实来自运行时,
  由 contract checker 与实验负责,不在本文件的背书范围内。
-/
import WitCert.SoftmaxTV
import Mathlib.Data.Real.Sqrt

open BigOperators Real

namespace WitCert.Calculus

variable {ι : Type*} [Fintype ι]

/-! ### ① 打分桥(Cauchy–Schwarz) -/

/--
  **打分桥**:score = scale·(q·k) 对条目扰动的敏感度(求和形式,与实现逐字对应)。
  右边两个因子都运行时可得:scale·√(∑q²) 由 qnorm 探针采,√(∑Δ²) 由带范数见证 W 上界。
-/
theorem score_bridge {d : ℕ} (q k k' : Fin d → ℝ) (scale : ℝ) (hs : 0 ≤ scale) :
    |scale * (∑ i, q i * k i) - scale * (∑ i, q i * k' i)|
      ≤ scale * Real.sqrt (∑ i, q i ^ 2) * Real.sqrt (∑ i, (k i - k' i) ^ 2) := by
  have hd : scale * (∑ i, q i * k i) - scale * (∑ i, q i * k' i)
      = scale * ∑ i, q i * (k i - k' i) := by
    rw [← mul_sub, ← Finset.sum_sub_distrib]
    congr 1
    exact Finset.sum_congr rfl fun i _ => by ring
  have hcs_pos : ∑ i, q i * (k i - k' i)
      ≤ Real.sqrt (∑ i, q i ^ 2) * Real.sqrt (∑ i, (k i - k' i) ^ 2) :=
    Real.sum_mul_le_sqrt_mul_sqrt _ _ _
  have hcs_neg : -(∑ i, q i * (k i - k' i))
      ≤ Real.sqrt (∑ i, q i ^ 2) * Real.sqrt (∑ i, (k i - k' i) ^ 2) := by
    have h := Real.sum_mul_le_sqrt_mul_sqrt Finset.univ q (fun i => -(k i - k' i))
    have he : ∑ i, q i * -(k i - k' i) = -(∑ i, q i * (k i - k' i)) := by
      rw [← Finset.sum_neg_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    have hsq : ∑ i, (-(k i - k' i)) ^ 2 = ∑ i, (k i - k' i) ^ 2 :=
      Finset.sum_congr rfl fun i _ => by ring
    rw [he, hsq] at h
    exact h
  rw [hd, abs_mul, abs_of_nonneg hs, mul_assoc]
  refine mul_le_mul_of_nonneg_left ?_ hs
  exact abs_le.mpr ⟨by linarith, hcs_pos⟩

/-! ### ② softmax 桥(tv_le_eform 的实例化) -/

/-- softmax(离散,全支撑)。 -/
noncomputable def softmax (s : ι → ℝ) : ι → ℝ :=
  fun t => Real.exp (s t) / ∑ u, Real.exp (s u)

lemma softmax_nonneg (s : ι → ℝ) (t : ι) : 0 ≤ softmax s t :=
  div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun _ _ => (Real.exp_pos _).le)

lemma sum_exp_pos [Nonempty ι] (s : ι → ℝ) : 0 < ∑ u, Real.exp (s u) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

lemma softmax_sum [Nonempty ι] (s : ι → ℝ) : ∑ t, softmax s t = 1 := by
  unfold softmax
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (sum_exp_pos s))

/-- 逐项恒等式:p̃_t·e^{s'_t − s_t} = e^{s'_t}/Z_s —— 倾斜的代数核心,单独立引理。 -/
lemma softmax_tilt (s s' : ι → ℝ) (t : ι) :
    softmax s t * Real.exp (-(s t - s' t)) = Real.exp (s' t) / ∑ u, Real.exp (s u) := by
  unfold softmax
  rw [div_mul_eq_mul_div, ← Real.exp_add]
  congr 2
  ring

/--
  **softmax TV 桥**:分数逐项扰动不超过 ε(ε ≥ 0),则分布全变差 ≤ ½(e^{2ε} − 1)。

  证明是**实例化而非另证**:取 p̃ = softmax s、p = softmax s'、ε_t = s t − s' t、
  c ≡ ε,则 p 恰是 p̃ 的 e^{−ε_t}/Z 倾斜(Z = Z_{s'}/Z_s),A = E_p̃[e^ε] = e^ε,
  论文1 的 tv_le_eform 直接给出 TV ≤ ½(A²−1) = ½(e^{2ε}−1)。
-/
theorem softmax_tv_bridge [Nonempty ι] (s s' : ι → ℝ) (ε : ℝ) (hε0 : 0 ≤ ε)
    (h : ∀ t, |s t - s' t| ≤ ε) :
    WitCert.TV (softmax s') (softmax s) ≤ (1/2) * (Real.exp ε ^ 2 - 1) := by
  have hZs := sum_exp_pos s
  have hZs' := sum_exp_pos s'
  -- Znorm (softmax s) (s − s') = Z_{s'} / Z_s
  have hZnorm : WitCert.Znorm (softmax s) (fun t => s t - s' t)
      = (∑ u, Real.exp (s' u)) / ∑ u, Real.exp (s u) := by
    unfold WitCert.Znorm
    calc ∑ t, softmax s t * Real.exp (-(s t - s' t))
        = ∑ t, Real.exp (s' t) / ∑ u, Real.exp (s u) :=
          Finset.sum_congr rfl fun t _ => softmax_tilt s s' t
      _ = (∑ t, Real.exp (s' t)) / ∑ u, Real.exp (s u) := by rw [Finset.sum_div]
  have hZpos : 0 < WitCert.Znorm (softmax s) (fun t => s t - s' t) := by
    rw [hZnorm]; positivity
  -- Acert (softmax s) (c ≡ ε) = e^ε
  have hA : WitCert.Acert (softmax s) (fun _ => ε) = Real.exp ε := by
    unfold WitCert.Acert
    rw [← Finset.sum_mul, softmax_sum, one_mul]
  have := WitCert.tv_le_eform (softmax s') (softmax s)
    (fun t => s t - s' t) (fun _ => ε)
    (softmax_nonneg s) (softmax_sum s)
    (fun _ => hε0) h hZpos
    (by
      intro t
      show softmax s' t = softmax s t * Real.exp (-(s t - s' t))
          / WitCert.Znorm (softmax s) (fun t => s t - s' t)
      rw [hZnorm, softmax_tilt s s' t]
      unfold softmax
      rw [div_div_div_cancel_right₀]
      exact ne_of_gt hZs)
  rwa [hA] at this

/-! ### ③ 仿射松弛(非线性界进仿射链的许可证) -/

/-- `e^x − 1 ≤ x·e^x`(x ≥ 0):由 `1 − x ≤ e^{−x}` 两边乘 `e^x`。 -/
lemma exp_sub_one_le_mul_exp {x : ℝ} (_hx : 0 ≤ x) :
    Real.exp x - 1 ≤ x * Real.exp x := by
  have h := Real.add_one_le_exp (-x)          -- −x + 1 ≤ e^{−x}
  have hpos := Real.exp_pos x
  have hmul : (-x + 1) * Real.exp x ≤ Real.exp (-x) * Real.exp x :=
    mul_le_mul_of_nonneg_right h hpos.le
  rw [← Real.exp_add, neg_add_cancel, Real.exp_zero] at hmul
  nlinarith

/--
  **仿射松弛**:0 ≤ ε ≤ ε₀ 时 ½(e^{2ε} − 1) ≤ e^{2ε₀}·ε。

  这是"softmax 桥"以仿射契约 (a, b) = (e^{2ε₀}, 0) 进链的许可证;
  前提 ε ≤ ε₀ 必须由运行时检查 —— 这正是该契约档位为 partial 的原因。
-/
theorem tv_affine_relax {ε ε₀ : ℝ} (h0 : 0 ≤ ε) (h1 : ε ≤ ε₀) :
    (1/2) * (Real.exp ε ^ 2 - 1) ≤ Real.exp (2 * ε₀) * ε := by
  have h2 : Real.exp ε ^ 2 = Real.exp (2 * ε) := by
    rw [pow_two, ← Real.exp_add]; ring_nf
  rw [h2]
  have h3 : Real.exp (2 * ε) - 1 ≤ (2 * ε) * Real.exp (2 * ε) :=
    exp_sub_one_le_mul_exp (by linarith)
  have h4 : Real.exp (2 * ε) ≤ Real.exp (2 * ε₀) :=
    Real.exp_le_exp.mpr (by linarith)
  nlinarith [Real.exp_pos (2 * ε), Real.exp_pos (2 * ε₀)]

/--
  **端到端桥(打包版,供论文引用)**:分数扰动 ε ∈ [0, ε₀] 时
      TV(softmax s', softmax s) ≤ e^{2ε₀} · ε。
  与 ① 相接:ε = scale·‖q‖·W,系数全部运行时可采。
-/
theorem softmax_tv_affine [Nonempty ι] (s s' : ι → ℝ) (ε ε₀ : ℝ)
    (hε0 : 0 ≤ ε) (hεb : ε ≤ ε₀) (h : ∀ t, |s t - s' t| ≤ ε) :
    WitCert.TV (softmax s') (softmax s) ≤ Real.exp (2 * ε₀) * ε :=
  le_trans (softmax_tv_bridge s s' ε hε0 h) (tv_affine_relax hε0 hεb)

end WitCert.Calculus
