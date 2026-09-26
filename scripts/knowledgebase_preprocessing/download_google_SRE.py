import os
import re
import requests
from bs4 import BeautifulSoup
import html2text
from tqdm import tqdm

# 保存路径
OUTPUT_DIR = "D:\projects\X-RAG\data\KnowledgeBase_raw\google_sre_workbook"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://sre.google"
TOC_URL = "https://sre.google/workbook/table-of-contents/"

# 配置 HTML 转 Markdown 工具
h = html2text.HTML2Text()
h.ignore_links = False
h.ignore_images = True
h.body_width = 0

print("1. 获取 SRE Book 目录列表...")
resp = requests.get(TOC_URL)
resp.encoding = 'utf-8'  # 👈 修复 1：目录页指定 UTF-8 编码
soup = BeautifulSoup(resp.text, 'html.parser')

# 查找所有章节链接
links = []
for a in soup.find_all('a', href=True):
    href = a['href']
    if href.startswith('/workbook/') and href != '/workbook/' and href != '/workbook/table-of-contents/':
        if href not in links:
            links.append(href)

print(f"共找到 {len(links)} 个章节，开始下载并转换为 Markdown...")

for idx, link in enumerate(tqdm(links, desc="Downloading")):
    chapter_url = BASE_URL + link
    c_resp = requests.get(chapter_url)
    if c_resp.status_code != 200:
        continue

    # 👈 修复 2：强行指定 UTF-8 解码，解决 can’t 变成 canâ€™t 的问题
    c_resp.encoding = 'utf-8'

    c_soup = BeautifulSoup(c_resp.text, 'html.parser')

    # 提取文章主体部分
    content_div = c_soup.find('section', class_='content') or c_soup.find('main') or c_soup.body
    if not content_div:
        continue

    # 转为 Markdown
    md_text = h.handle(str(content_div))

    # 格式化文件名
    clean_title = link.strip('/').split('/')[-1]
    file_name = f"{idx+1:02d}_{clean_title}.md"
    file_path = os.path.join(OUTPUT_DIR, file_name)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(md_text)

print(f"\n下载完成！所有 Markdown 文件已保存至: {OUTPUT_DIR}")