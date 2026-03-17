#!/usr/bin/env python3
"""
将指定目录下所有 PDF 的每一页导出为 PNG。
用法: pip install pymupdf && python pdf_pages_to_png.py [目录]
默认目录为脚本所在目录。
"""
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("请先安装 PyMuPDF: pip install pymupdf", file=sys.stderr)
    sys.exit(1)


def pdf_pages_to_png(dir_path: Path, dpi: int = 150) -> list[str]:
    """将 dir_path 下所有 PDF 的每一页导出为 PNG，返回生成的 PNG 路径列表。"""
    dir_path = dir_path.resolve()
    out_paths = []
    for pdf_path in sorted(dir_path.glob("*.pdf")):
        name_stem = pdf_path.stem
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            png_path = dir_path / f"{name_stem}_page{i + 1}.png"
            pix.save(str(png_path))
            out_paths.append(str(png_path))
        doc.close()
    return out_paths


def main():
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path(__file__).parent
    if not target.is_dir():
        print(f"错误: 不是目录: {target}", file=sys.stderr)
        sys.exit(1)
    paths = pdf_pages_to_png(target)
    if not paths:
        print(f"在 {target} 下未找到 PDF 文件。")
        return
    print(f"已导出 {len(paths)} 张 PNG:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
