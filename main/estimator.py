# ---------------------------------------------------------------------------- #
# Component 2: Estimator Module
# ---------------------------------------------------------------------------- #
import numpy as np

class Estimator:
    """
    論文の核心であるRL推定アルゴリズムを実装するクラス。
    """
    def calc_direct_RL_estimate(self,
                                chi_hat_ij_i_k:np.ndarray,
                                noisy_v:np.ndarray,
                                noisy_d:float,
                                noisy_d_dot:float,
                                T:float,
                                gamma:float
                                ) -> np.ndarray:
        """
        論文の式(1)に基づき、直接相対自己位置推定（Direct RL Estimation）を計算
        Args:
            current_estimate_x (np.ndarray): 現在の相対位置の推定値ベクトル (x̂_k)
            noisy_v (np.ndarray): ノイズを含む相対速度ベクトル (v_k + ε_k)
            noisy_d (float): ノイズを含む距離スカラー (d_k + ε_d)
            noisy_d_dot (float): ノイズを含む距離変化率スカラー (ḋ_k + ε_ḋ)
            T (float): サンプリング周期
            gamma (float): ゲインパラメータ (γ)
        Returns:
            np.ndarray: 次の時刻の相対位置の推定値ベクトル (x̂_{k+1})
        """
        # 式(1)を構成要素に分解
        # 第1項：現在の推定値
        current_RL_term = chi_hat_ij_i_k

        # 第2項：速度に基づく予測項
        predicton_term = T * noisy_v

        # 第3項：観測誤差に基づく補正項
        # 角括弧[]内のスカラー誤差を計算
        scalar_error = (noisy_d * noisy_d_dot) - (noisy_v.T @ chi_hat_ij_i_k)
        # スカラー誤差を用いてベクトル補正項を計算
        correction_term = gamma * T * noisy_v * scalar_error

        # 全ての項を結合して次の推定値を算出
        chi_hat_ij_i_k_plus_1 = current_RL_term + predicton_term + correction_term

        return chi_hat_ij_i_k_plus_1
