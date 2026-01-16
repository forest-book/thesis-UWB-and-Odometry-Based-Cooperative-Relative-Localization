import numpy as np
import ast
import operator
import types
from typing import List, Dict, Optional, Union
from collections import defaultdict
from enum import Enum, auto

class SafeExpressionEvaluator:
    """
    安全な数式評価器
    eval()の代わりにASTを使用して、許可された操作のみを実行する
    """
    # 許可された演算子
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }
    
    # 許可された関数
    ALLOWED_FUNCTIONS = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'sqrt': np.sqrt,
        'abs': abs
    }
    
    # 許可されたnumpy属性関数
    ALLOWED_NP_FUNCTIONS = {
        'sin', 'cos', 'tan', 'sqrt', 'abs',
        'arcsin', 'arccos', 'arctan', 'arctan2',
        'exp', 'log', 'log10', 'power', 'square'
    }
    
    def __init__(self, allowed_names: Dict[str, any]):
        """
        Parameters:
        -----------
        allowed_names : dict
            許可された変数名とその値の辞書 (例: {'k': 0.0, 'np': np})
        """
        self.allowed_names = allowed_names
    
    def eval(self, expression: str) -> Union[float, int, np.ndarray]:
        """
        安全に数式を評価する
        
        Parameters:
        -----------
        expression : str
            評価する数式文字列
            
        Returns:
        --------
        Union[float, int, np.ndarray]
            評価結果
        """
        try:
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body)
        except ValueError:
            # ValueErrorは既に適切なエラーメッセージを持っているので再スロー
            raise
        except Exception:
            # その他の例外は詳細を隠してセキュリティを確保
            raise ValueError(f"式の評価中にエラーが発生しました: {expression}")
    
    def _eval_node(self, node: ast.AST):
        """ASTノードを再帰的に評価する
        
        Parameters:
        -----------
        node : ast.AST
            評価するASTノード
            
        Returns:
        --------
        Union[float, int, np.ndarray]
            評価結果
        """
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in self.allowed_names:
                return self.allowed_names[node.id]
            else:
                raise ValueError(f"許可されていない変数: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                return self.ALLOWED_OPERATORS[op_type](left, right)
            else:
                raise ValueError(f"許可されていない演算子: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                return self.ALLOWED_OPERATORS[op_type](operand)
            else:
                raise ValueError(f"許可されていない単項演算子: {op_type}")
        elif isinstance(node, ast.Call):
            # 関数呼び出しの処理
            if isinstance(node.func, ast.Attribute):
                # np.sin(x) のような属性アクセスの場合
                # まず、属性の名前を取得（評価前）
                if not isinstance(node.func.value, ast.Name):
                    raise ValueError("許可されていない複雑な属性アクセス")
                
                obj_name = node.func.value.id
                func_name = node.func.attr
                
                # numpy モジュールのみ許可
                if obj_name == "np" and obj_name in self.allowed_names:
                    if func_name in self.ALLOWED_NP_FUNCTIONS:
                        obj = self.allowed_names[obj_name]
                        # セキュリティチェック: numpyモジュールであることを確認
                        if obj is not np or not isinstance(obj, types.ModuleType):
                            raise ValueError("不正なnumpyオブジェクト")
                        func = getattr(obj, func_name)
                        args = [self._eval_node(arg) for arg in node.args]
                        return func(*args)
                    else:
                        raise ValueError(f"許可されていないnumpy関数: np.{func_name}")
                else:
                    raise ValueError(f"許可されていない属性アクセス: {obj_name}.{func_name}")
            elif isinstance(node.func, ast.Name):
                # sin(x) のような直接関数呼び出しの場合
                func_name = node.func.id
                if func_name in self.ALLOWED_FUNCTIONS:
                    func = self.ALLOWED_FUNCTIONS[func_name]
                    args = [self._eval_node(arg) for arg in node.args]
                    return func(*args)
                else:
                    raise ValueError(f"許可されていない関数: {func_name}")
            else:
                raise ValueError("許可されていない関数呼び出し")
        else:
            raise ValueError(f"許可されていないノードタイプ: {type(node).__name__}")

class Scenario(Enum):
    CONTINUOUS = auto()
    SUDDEN_TURN = auto()

class UAV:
    """
    各UAVの状態と機能を管理するクラス
    論文 V-A-1節「Configuration」に基づき、UAVのダイナミクスを定義
    """
    def __init__(self, uav_id: int,
                 initial_position: np.ndarray,
                 neighbors: List[int],
                 trajectory_config: Optional[Dict[str, str]] = None):
        self.id: int = uav_id
        self.true_position = np.array(initial_position, dtype=float)
        self.neighbors: List[int] = neighbors
        self.trajectory_config = trajectory_config

        # 速度式を保持
        self.vx_formula = None
        self.vy_formula = None
        
        # 安全な評価器を作成
        self.safe_evaluator = SafeExpressionEvaluator({
            "np": np,
            "k": 0.0
        })

        if self.trajectory_config and self.trajectory_config.get('type') == 'formula':
            self.vx_formula = self.trajectory_config['vx']
            self.vy_formula = self.trajectory_config['vy']

        # 初期速度の計算
        self.update_velocity(t=0, dt=0)  # dtは初期化時は0でOK

        # 推定値を保持する辞書 {target_id: estimate_vector}
        self.direct_estimates: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.fused_estimates: Dict[str, List[np.ndarray]] = defaultdict(list)

    def update_velocity(self, t: int, dt: float):
        """現在の時刻kに基づいて速度ベクトルを更新"""
        # 注: 添え字のkは離散時間ステップだが，速度式内部のkは実時間であるみたい
        # 速度は [m/s] 単位として解釈し、dt を掛けて位置を更新
        k = t * dt  # 速度式内部のkなので実時間に変換

        if self.vx_formula and self.vy_formula:
            # 安全な評価器のコンテキストを更新
            self.safe_evaluator.allowed_names["k"] = k
            try:
                vx = self.safe_evaluator.eval(self.vx_formula)
                vy = self.safe_evaluator.eval(self.vy_formula)
                self.true_velocity = np.array([vx, vy], dtype=float)
            except Exception as e:
                print(f"Error calculating trajectory for UAV {self.id}: {e}")
                raise
        else:
            self.true_velocity = np.zeros(2)

    def update_state(self, t: int, dt: float, event: Scenario = Scenario.CONTINUOUS):
        """UAVの真の位置と速度を更新する"""
        k = t * dt  # 速度式内部のkなので実時間に変換

        # 速度の更新
        self.update_velocity(t=t, dt=dt)

        # シナリオ2: UAV4の急な機動変更イベント
        if self.id == 4 and event == Scenario.SUDDEN_TURN and 100 <= k < 101:
            self.true_velocity += np.array([5.0, 5.0]) # 外乱を追加

        # 位置の更新: v [m/s] × dt [s] = 変位 [m]
        self.true_position += self.true_velocity * dt
