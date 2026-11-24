"""tools/device_a02.sh 的测试：verdict 解读层 + 前置检查。

为什么这层值得单独测：device_a02.sh 里只有两片是纯逻辑，其余（cmake、跑探针）
只能在设备上验。这两片分别是：

  1. verdict 字符串 → 退出码 + 下一步建议
  2. 前置检查的判定 —— 尤其"图 API 有默认值"这一条

第 2 条尤其要紧：graph_compat.h 用 `#ifndef` 给了默认 1，所以**任何 SoC 上都编译得过**。
如果前置检查把"取到了值"当成"已确认"，脚本就会对纯猜测打绿勾，人看到绿勾就跳过 A-01。
这种假绿比报错贵得多，所以这里对它单独设一条断言。

而这恰恰是最容易错的地方：`content_folded` 和 `kernel_broken` 的处理方式完全相反 ——
前者要改架构，后者要修 kernel。把两者读反，会让人拿一个坏 kernel 当证据去推翻
ADR-005，那是全项目最贵的一种错误。

所以这里对七种 verdict 逐一断言退出码，并抽查关键结论句没被写反。
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "device_a02.sh"

# 退出码契约（与 device_a02.sh 头部注释一致）：
EXIT_PASS = 0        # indirection_supported
EXIT_FAIL_VERDICT = 4  # 有效结果，但是失败态
EXIT_MALFORMED = 3   # 结果本身不完整／不可识别
EXIT_NO_FILE = 1     # 文件不存在


def run_interpret(payload: dict | None, raw: str | None = None) -> tuple[int, str]:
    """把一份 JSON 喂给 --interpret，返回 (退出码, 输出)。"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        if raw is not None:
            fh.write(raw)
        else:
            json.dump(payload, fh)
        path = fh.name
    try:
        proc = subprocess.run(
            [str(SCRIPT), "--interpret", path],
            capture_output=True, text=True, timeout=60,
        )
        return proc.returncode, proc.stdout + proc.stderr
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


class InterpretTest(unittest.TestCase):
    def test_script_is_executable(self):
        self.assertTrue(SCRIPT.exists(), f"{SCRIPT} 不存在")
        self.assertTrue(SCRIPT.stat().st_mode & 0o111, "脚本没有执行位")

    def test_pass_verdict_returns_zero(self):
        rc, out = run_interpret({"verdict": "indirection_supported", "remedy": "proceed"})
        self.assertEqual(rc, EXIT_PASS, out)
        self.assertIn("间接寻址成立", out)

    def test_kernel_broken_insists_it_is_not_an_architecture_conclusion(self):
        """最容易被读错的一种：kernel 坏了不是架构结论。

        如果在输出里把它说成架构判决，人类就会去改 DESIGN——而真正要改的是 kernel。
        """
        rc, out = run_interpret({"verdict": "kernel_broken", "remedy": "fix kernel"})
        self.assertEqual(rc, EXIT_FAIL_VERDICT, out)
        self.assertIn("不是架构结论", out)

    def test_content_folded_forbids_patching_the_code(self):
        """折叠是唯一会推翻主路径的结论，且必须明确禁止"改代码绕过去"。"""
        rc, out = run_interpret({"verdict": "content_folded", "remedy": "switch to A' / B / C"})
        self.assertEqual(rc, EXIT_FAIL_VERDICT, out)
        self.assertIn("不要改代码去绕", out)
        self.assertIn("ADR-005", out)

    def test_empty_graph_does_not_jump_to_the_fallback(self):
        """empty_graph 通常是捕获区间没包住，不是"图不可捕获"。

        直接转后备探针会浪费三天，所以输出必须先怀疑捕获区间。
        """
        rc, out = run_interpret({"verdict": "empty_graph", "remedy": "switch to A'"})
        self.assertEqual(rc, EXIT_FAIL_VERDICT, out)
        self.assertIn("捕获区间", out)

    def test_first_replay_stale_points_at_h2d_ordering(self):
        rc, out = run_interpret({"verdict": "first_replay_stale"})
        self.assertEqual(rc, EXIT_FAIL_VERDICT, out)
        self.assertIn("H2D", out)

    def test_remaining_failure_verdicts_all_return_four(self):
        for name in ("capture_path_broken", "replay_not_executing"):
            with self.subTest(verdict=name):
                rc, out = run_interpret({"verdict": name})
                self.assertEqual(rc, EXIT_FAIL_VERDICT, f"{name}: {out}")

    def test_unknown_verdict_is_malformed_not_a_verdict(self):
        """陌生字符串意味着探针与脚本版本不一致 —— 那是工具问题，不是架构结论。"""
        rc, out = run_interpret({"verdict": "something_new"})
        self.assertEqual(rc, EXIT_MALFORMED, out)
        self.assertIn("未知 verdict", out)

    def test_missing_verdict_field_is_malformed(self):
        rc, out = run_interpret({"arms": []})
        self.assertEqual(rc, EXIT_MALFORMED, out)

    def test_unreadable_json_is_malformed(self):
        rc, out = run_interpret(None, raw="{not json")
        self.assertEqual(rc, EXIT_MALFORMED, out)

    def test_nonexistent_file_fails(self):
        proc = subprocess.run(
            [str(SCRIPT), "--interpret", "/nonexistent/nope.json"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertNotEqual(proc.returncode, 0)

    def test_interpret_works_without_any_cann_environment(self):
        """解读层必须能在无卡、无 CANN 的机器上工作 —— 整个 host CI 依赖这一点。"""
        env_free = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(pathlib.Path.home()),
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"verdict": "indirection_supported"}, fh)
            path = fh.name
        try:
            proc = subprocess.run(
                [str(SCRIPT), "--interpret", path],
                capture_output=True, text=True, timeout=60, env=env_free,
            )
            self.assertEqual(proc.returncode, EXIT_PASS, proc.stdout + proc.stderr)
        finally:
            pathlib.Path(path).unlink(missing_ok=True)


class PreflightTest(unittest.TestCase):
    """前置检查：在无卡机器上就能全部验证。"""

    @staticmethod
    def run_script(args: list[str], env: dict[str, str] | None = None):
        base = {"PATH": "/usr/bin:/bin", "HOME": str(pathlib.Path.home())}
        if env:
            base.update(env)
        return subprocess.run(
            [str(SCRIPT), *args], capture_output=True, text=True, timeout=120, env=base,
        )

    def test_missing_ascend_home_blocks_the_run(self):
        rc = self.run_script(["--dry-run", "--soc", "ascend910_9391", "--graph-api", "1"])
        self.assertEqual(rc.returncode, 1, rc.stdout + rc.stderr)
        self.assertIn("ASCEND_HOME_PATH", rc.stdout)
        self.assertIn("不要往下走", rc.stdout)

    def test_default_graph_api_is_reported_as_a_guess_not_a_result(self):
        """核心断言：默认值必须被说成"不是 A-01 的结论"，不能是绿勾。

        这一条如果失效，脚本会让人以为 A-01 已经做完。
        """
        rc = self.run_script(["--dry-run", "--soc", "ascend910_9391"])
        combined = rc.stdout + rc.stderr
        self.assertIn("这不是 A-01 的结论", combined)
        # 且不能出现"默认值"加绿勾的组合说法
        self.assertNotIn("✓ ANYINFER_GRAPH_API = 1", combined)

    def test_explicit_graph_api_is_accepted(self):
        rc = self.run_script(
            ["--dry-run", "--soc", "ascend910_9391", "--graph-api", "2"],
            env={"ASCEND_HOME_PATH": tempfile.gettempdir()},
        )
        combined = rc.stdout + rc.stderr
        self.assertIn("显式指定", combined)
        self.assertIn("-DANYINFER_GRAPH_API=2", combined)

    def test_bad_graph_api_value_is_rejected(self):
        rc = self.run_script(
            ["--dry-run", "--soc", "ascend910_9391", "--graph-api", "3"],
            env={"ASCEND_HOME_PATH": tempfile.gettempdir()},
        )
        self.assertEqual(rc.returncode, 1, rc.stdout + rc.stderr)
        self.assertIn("只能是 1 或 2", rc.stdout)

    def test_happy_path_dry_run_reaches_the_build_step(self):
        """前置检查全过时，dry-run 应打印 cmake 命令并停在编译前（不真的编译）。"""
        rc = self.run_script(
            ["--dry-run", "--soc", "ascend910_9391", "--graph-api", "1"],
            env={"ASCEND_HOME_PATH": tempfile.gettempdir()},
        )
        combined = rc.stdout + rc.stderr
        self.assertEqual(rc.returncode, 0, combined)
        self.assertIn("ANYINFER_WITH_CANN=ON", combined)
        self.assertIn("ascend910_9391", combined)
        self.assertIn("不实际执行", combined)

    def test_unknown_flag_is_rejected(self):
        rc = self.run_script(["--nope"])
        self.assertEqual(rc.returncode, 64, rc.stdout + rc.stderr)


if __name__ == "__main__":
    if not SCRIPT.exists():
        print(f"跳过：{SCRIPT} 不存在", file=sys.stderr)
        sys.exit(1)
    unittest.main()
