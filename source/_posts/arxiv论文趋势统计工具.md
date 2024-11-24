---
title: arxiv论文趋势统计工具
date: 2024-10-30 21:11:39
tags:
  - 技术
  - 自学
categories: 大学
cover: https://ooo.0x0.ooo/2024/10/30/ODwPf1.jpg
---

## 展示

（和 GPT）写了个小组件来查关键词在 arxiv 的论文发布趋势，用 ploty.js 写的前端，bs4 写的后端，感觉还行，等个时间整理了传在 github 上。

<iframe src="/custom-page/arxiv.html" height="500px" width="100%" scrolling="auto" frameborder="0" style="">
</iframe>

## 粗糙源码

### 前端

#### React 版本

```ts
import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Bar,
  BarChart,
  ComposedChart,
} from "recharts";

const TrendVisualization = () => {
  const data = [
    { month: "2022-09", papers: 1 },
    { month: "2023-02", papers: 1 },
    { month: "2023-03", papers: 1 },
    { month: "2023-07", papers: 1 },
    { month: "2023-08", papers: 5 },
    { month: "2023-09", papers: 2 },
    { month: "2023-10", papers: 2 },
    { month: "2023-11", papers: 2 },
    { month: "2023-12", papers: 1 },
    { month: "2024-02", papers: 2 },
    { month: "2024-03", papers: 1 },
    { month: "2024-04", papers: 2 },
    { month: "2024-05", papers: 3 },
    { month: "2024-06", papers: 3 },
    { month: "2024-07", papers: 2 },
    { month: "2024-08", papers: 5 },
    { month: "2024-09", papers: 2 },
    { month: "2024-10", papers: 8 },
  ];

  // 计算 3 个月移动平均
  const calculateMovingAverage = (data, window = 3) => {
    return data.map((item, index) => {
      const start = Math.max(0, index - window + 1);
      const values = data.slice(start, index + 1).map((d) => d.papers);
      const avg = values.reduce((a, b) => a + b, 0) / values.length;
      return {
        ...item,
        movingAverage: parseFloat(avg.toFixed(2)),
      };
    });
  };

  const dataWithMA = calculateMovingAverage(data);

  return (
    <Card className="w-full max-w-4xl">
      <CardHeader>
        <CardTitle>Graph Prompting Research Trend</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={dataWithMA}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="month"
                angle={-45}
                textAnchor="end"
                interval={0}
                height={60}
              />
              <YAxis
                label={{
                  value: "Number of Papers",
                  angle: -90,
                  position: "insideLeft",
                  offset: -5,
                }}
              />
              <Tooltip />
              <Legend />
              <Bar
                dataKey="papers"
                fill="#8884d8"
                name="Monthly Papers"
                barSize={20}
              />
              <Line
                type="monotone"
                dataKey="movingAverage"
                stroke="#ff7300"
                name="3-Month Moving Average"
                strokeWidth={2}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};
export default TrendVisualization;
```

#### html 版本

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Graph Prompting Research Trend</title>
    <script
      src="https://cdn.plot.ly/plotly-2.35.2.min.js"
      charset="utf-8"
    ></script>
    <style>
      body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
      }
      .card {
        width: 100%;
        max-width: 800px;
        margin: auto;
        border: 1px solid #ccc;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      }
      .card-header {
        background: #f5f5f5;
        padding: 16px;
        font-size: 1.5em;
        font-weight: bold;
      }
      .card-content {
        padding: 0;
      }
      .chart-container {
        width: 100%;
        height: 400px;
      }
    </style>
  </head>
  <body>
    <div class="card">
      <div class="card-header">Graph Prompting Research Trend</div>
      <div class="card-content">
        <div id="chart" class="chart-container"></div>
      </div>
    </div>

    <script>
      const data = [
        { month: "2022-09", papers: 1 },
        { month: "2023-02", papers: 1 },
        { month: "2023-03", papers: 1 },
        { month: "2023-07", papers: 1 },
        { month: "2023-08", papers: 5 },
        { month: "2023-09", papers: 2 },
        { month: "2023-10", papers: 2 },
        { month: "2023-11", papers: 2 },
        { month: "2023-12", papers: 1 },
        { month: "2024-02", papers: 2 },
        { month: "2024-03", papers: 1 },
        { month: "2024-04", papers: 2 },
        { month: "2024-05", papers: 3 },
        { month: "2024-06", papers: 3 },
        { month: "2024-07", papers: 2 },
        { month: "2024-08", papers: 5 },
        { month: "2024-09", papers: 2 },
        { month: "2024-10", papers: 8 },
      ];

      // Calculate cumulative papers
      const cumulativeData = data.map((item, index) => ({
        month: item.month,
        cumulativePapers: data
          .slice(0, index + 1)
          .reduce((acc, cur) => acc + cur.papers, 0),
        papers: item.papers,
      }));

      const trace1 = {
        x: cumulativeData.map((d) => d.month),
        y: cumulativeData.map((d) => d.cumulativePapers),
        type: "bar",
        name: "Cumulative Papers",
        marker: { color: "#8884d8" },
      };

      const trace2 = {
        x: data.map((d) => d.month),
        y: data.map((d) => d.papers),
        mode: "lines+markers",
        name: "Monthly Papers",
        line: { color: "#ff7300", width: 2 },
      };

      const layout = {
        title: "Graph Prompting Research Trend",
        xaxis: {
          title: "Month",
          tickangle: -45,
        },
        yaxis: {
          title: "Number of Papers",
          titlefont: { size: 16 },
          autorange: true,
        },
        barmode: "overlay",
      };

      Plotly.newPlot("chart", [trace1, trace2], layout);
    </script>
  </body>
</html>
```

### 后端

```python
import urllib.request as libreq
import xml.etree.ElementTree as ET
import pandas as pd
import time
from datetime import datetime
import csv
from collections import Counter
import re

def create_search_query(search_terms, mode='broad'):
    """
    创建不同模式的搜索查询
    """
    if mode == 'strict':
        return f'%22{search_terms}%22'
    elif mode == 'medium':
        return f'({search_terms})'
    else:
        return f'({search_terms})'

def fetch_papers(search_terms, mode='broad', start=0, max_results=100):
    """
    获取arXiv论文数据
    """
    query = create_search_query(search_terms, mode)
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'

    try:
        with libreq.urlopen(url) as response:
            r = response.read()
        return ET.fromstring(r)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def check_relevance(text, search_terms, mode='broad'):
    """
    检查文本是否相关，支持不同的匹配模式
    """
    text = text.lower()

    # 定义不同严格程度的关键词组合
    strict_patterns = [
        rf'{search_terms}'
    ]

    medium_patterns = strict_patterns + [
        rf'{search_terms.replace(" ", ".*")}'
    ]

    broad_patterns = medium_patterns + [
        rf'{search_terms.replace(" ", ".*")}'
    ]

    # 选择对应模式的模式列表
    patterns = (strict_patterns if mode == 'strict' else
               medium_patterns if mode == 'medium' else
               broad_patterns)

    # 检查是否匹配任何模式
    return any(re.search(pattern, text) for pattern in patterns)

def parse_entry(entry, search_terms, search_mode='broad'):
    """
    解析单个论文条目，增加相关性评分
    """
    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    title = entry.find('atom:title', ns).text.strip()
    summary = entry.find('atom:summary', ns).text.strip()
    published = entry.find('atom:published', ns).text

    date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ')
    year_month = f"{date.year}-{date.month:02d}"
    full_date = date.strftime('%Y-%m-%d')

    authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
    categories = [cat.get('term') for cat in entry.findall('atom:category', ns)]

    text = f"{title} {summary}"
    is_relevant = check_relevance(text, search_terms, search_mode)

    relevance_score = 1 if is_relevant else 0
    if 'cs.AI' in categories or 'cs.LG' in categories:
        relevance_score += 0.5

    return {
        'title': title,
        'summary': summary,
        'year_month': year_month,
        'full_date': full_date,
        'authors': '; '.join(authors),
        'categories': '; '.join(categories),
        'is_relevant': is_relevant,
        'relevance_score': relevance_score
    }

def collect_all_papers(search_terms, search_mode='broad'):
    """
    收集所有相关论文数据
    """
    papers = []
    start = 0
    batch_size = 100
    total_checked = 0

    while True:
        print(f"Fetching papers from index {start}")
        root = fetch_papers(search_terms, search_mode, start=start, max_results=batch_size)

        if root is None:
            break

        entries = root.findall('{http://www.w3.org/2005/Atom}entry')

        if not entries:
            break

        for entry in entries:
            paper_data = parse_entry(entry, search_terms, search_mode)
            total_checked += 1

            if paper_data['is_relevant']:
                papers.append(paper_data)

        print(f"Checked {total_checked} papers, found {len(papers)} relevant papers...")

        time.sleep(3)
        start += batch_size

        if len(entries) < batch_size:
            break

    return papers

def save_results(papers, search_terms, search_mode, base_filename='results'):
	"""
	保存结果并生成详细报告
	"""
	if not papers:
		print("No papers found!")
		return None, None

	df = pd.DataFrame(papers)

	detailed_filename = f'{base_filename}_{search_terms[:5]}_{search_mode}_detailed.csv'
	df.to_csv(detailed_filename, index=False, encoding='utf-8')

	stats = {
		'total_papers': len(df),
		'by_category': df['categories'].str.get_dummies(sep='; ').sum().to_dict(),
		'by_month': df.groupby('year_month').size().to_dict(),
		'avg_relevance': df['relevance_score'].mean(),
	}

	report_filename = f'{base_filename}_{search_terms[:5]}_{search_mode}_report.txt'
	with open(report_filename, 'w', encoding='utf-8') as f:
		f.write(f"Research Report for {search_terms} ({search_mode} mode)\n")
		f.write("="* 50 + "\n\n")
		f.write(f"Total Papers Found: {stats['total_papers']}\n\n")

		f.write("Distribution by Category:\n")
		for cat, count in stats['by_category'].items():
			f.write(f"{cat}: {count}\n")

		f.write("\nMonthly Distribution:\n")
		for month, count in sorted(stats['by_month'].items()):
			f.write(f"{month}: {count}\n")

		f.write(f"\nAverage Relevance Score: {stats['avg_relevance']:.2f}\n")

	return df, stats

if __name__ == "__main__":
    search_terms = 'resnet'
    modes = ['strict']
    # modes = ['strict', 'medium', 'broad']
    results = {}

    for mode in modes:
        print(f"\nRunning {mode} search mode...")
        papers = collect_all_papers(search_terms, mode)
        df, stats = save_results(papers, search_terms, mode)
        results[mode] = {
            'papers': papers,
            'stats': stats
        }

        print(f"\n{mode.capitalize()} mode results:")
        print(f"Total papers found: {len(papers)}")
        if stats:
            print(f"Average relevance score: {stats['avg_relevance']:.2f}")
```

## 结语

内嵌这个 html 在网站里面还挺缺德的，hexo 不能直接插入 div 和 script 导致我自己是用 html 然后用 if-frame 插入的。

今天先这样，快乐结束！
