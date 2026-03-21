import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

def plot_network_graph(G, nodes_data):
    """
    使用 Plotly 绘制网络拓扑图
    """
    pos = nx.spring_layout(G, seed=42)
    
    # 1. 绘制边 (Edges)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # 2. 绘制节点 (Nodes)
    node_x = []
    node_y = []
    node_color = []
    node_text = []
    
    for node_idx in G.nodes():
        x, y = pos[node_idx]
        node_x.append(x)
        node_y.append(y)
        # 获取节点对应的颜色和文本
        node_info = nodes_data[node_idx]
        node_color.append(node_info.get('color', '#888'))
        
        # Safe access to node info
        nid = node_info.get('id', str(node_idx))
        name = node_info.get('name', 'Unknown')
        ntype = node_info.get('type', 'Unknown')
        
        # Check for 'text' key if explicitly provided, otherwise construct
        if 'text' in node_info:
             node_text.append(str(node_info['text']))
        else:
             node_text.append(f"ID: {nid}<br>昵称: {name}<br>类型: {ntype}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=15,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
                ))
    return fig

def plot_trend_chart(df):
    """
    绘制舆情趋势折线图
    """
    fig_trend = px.line(df, x="日期", y=["正常信息", "失序信息"], 
                        color_discrete_map={"正常信息": "#3498DB", "失序信息": "#E74C3C"},
                        markers=True)
    fig_trend.update_layout(xaxis_title="日期", yaxis_title="信息数量", legend_title="信息类型", hovermode="x unified")
    return fig_trend

def plot_pie_chart(data, values, names):
    """
    绘制饼图
    """
    fig_pie = px.pie(data, values=values, names=names, hole=0.4, 
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_pie.update_layout(legend_position="bottom")
    return fig_pie

def plot_radar_chart(data_dict, color='#8E44AD', range_r=None):
    """
    绘制雷达图
    """
    df_radar = pd.DataFrame(dict(
        r=list(data_dict.values()),
        theta=list(data_dict.keys())
    ))
    fig_radar = px.line_polar(df_radar, r='r', theta='theta', line_close=True, range_r=range_r)
    fig_radar.update_traces(fill='toself', line_color=color)
    fig_radar.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    return fig_radar

def plot_text_heatmap(words, scores, width=10, font_family='Times New Roman'):
    """
    使用 Matplotlib 绘制论文级文本热力图
    Args:
        words: List[str], 单词列表
        scores: List[float], 分数列表 (0.0 - 1.0)
        width: int, 每行大致单词数 (用于估算高度)
        font_family: str, 字体
    Returns:
        fig: matplotlib.figure.Figure
    """
    # 配置
    plt.rcParams['font.family'] = font_family
    cmap = plt.cm.Reds
    
    # --- 预计算排版 (紧凑估算) ---
    # 根据 width 参数（每行单词数）估算行数
    est_lines = (len(words) / width) * 1.0 + 1 # 减少冗余行
    
    # 动态计算 Figure 尺寸
    # 基础高度 + 每行增量
    # 增大每行的高度分配，适应更大的字体 (0.5 -> 0.7)
    fig_height = max(3, est_lines * 0.7) 
    
    # 创建 Figure
    fig = plt.figure(figsize=(12, fig_height)) # 宽度略微增加
    
    # 主绘图区 (紧贴边缘)
    # [left, bottom, width, height]
    # 留出极少的边距，例如 0.02
    margin = 0.02
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin]) 
    
    # 开启边框 but make it slightly thinner for academic style
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove borders (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # for spine in ax.spines.values():
    #     spine.set_visible(True)
    #     spine.set_linewidth(1.5)
    #     spine.set_color('#1a1a1a') # 接近黑色的深灰

    # 显式设置 Limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # 获取 renderer
    canvas = FigureCanvasAgg(fig)
    renderer = fig.canvas.get_renderer()
    
    # 初始坐标
    x_start = 0.02 # 更靠左
    # WRAP_WIDTH: 文本块的最大允许宽度 (不包含左边距)
    # 稍微放宽以利用更多空间
    wrap_width = 0.90 
    x_limit = x_start + wrap_width
    
    y_start = 0.95 # 更靠上
    x = x_start
    y = y_start
    
    # 动态行高
    # 稍微减小行距以更紧凑
    line_height_rel = 0.6 / fig_height
    
    # 记录实际的文本边界
    max_text_right = 0.0
    min_text_left = 1.0 # 用于计算实际起始位置
    
    # --- 隐形边界锚点 (Ghost Anchors) ---
    # Removed to prevent excessive whitespace in saved image
    # The axes spines (border) will define the bbox now.
    # ax.text(0, 1.02, ".", alpha=0)
    # ax.text(0, -0.02, ".", alpha=0)
    # ax.text(0, 0, ".", alpha=0)
    # ax.text(1, 1, ".", alpha=0)
    
    # 归一化 scores
    if len(scores) > 0 and np.max(scores) > 1.0:
        scores = np.array(scores) / np.max(scores)
        
    drawn_texts = [] # 存储所有文本对象以便后续移动
    
    for word, score in zip(words, scores):
        word = str(word) # Ensure string
        bg = cmap(score)
        txt_col = 'white' if score > 0.6 else 'black'
        
        # 试探性绘制
        # Academic style: Square/Sharp box
        t = ax.text(x, y, word, fontsize=26, color=txt_col,
                    bbox=dict(facecolor=bg, edgecolor='none', boxstyle='square,pad=0.2'),
                    verticalalignment='top', fontname=font_family)
        
        # 获取宽度
        bbox = t.get_window_extent(renderer=renderer)
        bbox_data = ax.transData.inverted().transform(bbox)
        width_data = bbox_data[1][0] - bbox_data[0][0]
        
        # 检查是否越界
        if x + width_data > x_limit:
            t.remove()
            # 换行
            x = x_start
            y -= line_height_rel
            # 重画
            # Academic style: Square/Sharp box
            t = ax.text(x, y, word, fontsize=26, color=txt_col,
                        bbox=dict(facecolor=bg, edgecolor='none', boxstyle='square,pad=0.2'),
                        verticalalignment='top', fontname=font_family)
            # 更新宽度
            bbox = t.get_window_extent(renderer=renderer)
            bbox_data = ax.transData.inverted().transform(bbox)
            width_data = bbox_data[1][0] - bbox_data[0][0]
            
        # 记录
        drawn_texts.append(t)
        
        # 更新最大右侧位置
        current_right = x + width_data
        if current_right > max_text_right:
            max_text_right = current_right
        if x < min_text_left:
            min_text_left = x
            
        x += width_data + 0.015 # 间距
        
    # --- 垂直居中调整 ---
    # 计算当前内容的实际垂直范围
    # 顶部是 y_start (0.90)
    # 底部是最后一个文本的 y - line_height_rel (估算)
    # 但由于 verticalalignment='top', 最后一行的 top 是 y.
    # 所以内容范围是 [y - line_height_rel, y_start] (roughly)
    # 为了更精确，我们可以遍历 drawn_texts 获取 bbox (但 bbox 是 transform 后的)
    # 简单起见，使用 y 指针。
    # 当前 y 指向最后一行文本的 top.
    
    # 实际内容底部 (给一点 margin)
    current_content_bottom = y - line_height_rel * 0.8
    current_content_top = y_start
    
    current_content_height = current_content_top - current_content_bottom
    current_content_mid = (current_content_top + current_content_bottom) / 2
    
    target_mid = 0.5
    v_shift = target_mid - current_content_mid
    
    # 应用垂直偏移
    for t in drawn_texts:
        pos = t.get_position()
        t.set_y(pos[1] + v_shift)
        
    # 更新 y_start 和 y 用于后续 Colorbar 计算
    # Colorbar 依赖 text_top 和 text_bottom
    # 我们更新这两个变量即可
    text_top = current_content_top + v_shift
    text_bottom = current_content_bottom + v_shift
    
    # --- 动态居中调整 (水平) ---
    # 计算内容总宽度：文本块实际宽度 + 间距 + Colorbar宽度
    # 额外预留 Colorbar Label 的宽度 (Label is vertical on the right)
    # Estimate: ~0.02 normalized units for tick labels to be safe (0.05 was tight)
    cbar_gap = 0.05
    cbar_width = 0.02
    cbar_label_allowance = 0.02 
    
    # Recalculate actual_text_width from x_start (0.05) since min_text_left is not updated by vertical shift
    # And we want to be explicit.
    # The text started at x_start (0.05).
    actual_text_width = max_text_right - x_start
    if actual_text_width < 0: actual_text_width = 0 # Empty case
    
    # Total visual width now includes the label allowance
    total_visual_width = actual_text_width + cbar_gap + cbar_width + cbar_label_allowance
    
    # 目标左边距 (Canvas 宽度为 1.0)
    target_left = (1.0 - total_visual_width) / 2
    
    # Ensure we don't start too far left (e.g. < 0.02 due to border)
    if target_left < 0.02: target_left = 0.02

    # --- Hard Right Boundary Check ---
    # Even if centered, we must ensure the rightmost visual element (label) doesn't cross 0.98
    # Estimated right edge = target_left + total_visual_width
    # We want right_edge <= 0.98
    
    estimated_right_edge = target_left + total_visual_width
    if estimated_right_edge > 0.98:
        # Force shift left to fit
        overshoot = estimated_right_edge - 0.98
        target_left -= overshoot
        
        # Re-check left boundary (if we are squeezed, we might hit left)
        if target_left < 0.02:
            target_left = 0.015

    # 计算偏移量 (Shift)
    # 我们当前的 min_text_left 是 x_start (0.05)
    shift = target_left - x_start
    
    # 应用偏移
    for t in drawn_texts:
        pos = t.get_position()
        t.set_x(pos[0] + shift)
        
    # 更新 max_text_right 以便放置 Colorbar
    max_text_right += shift
        
    # --- 最终布局优化：Resize & Crop ---
    # 计算实际内容的边界 (Text + Colorbar)
    # 我们希望 Figure 紧贴这个边界
    
    # 文本左边界 (shift 后的 min_text_left)
    # 实际上 min_text_left 在循环中更新的是 shift 前的，但我们已经计算了 shift
    # target_left 就是最终的最左边位置 (除了 0.02 的边界保护)
    # 但是我们添加了 margin=0.02 在 axes 上。
    # ax limits 是 0-1.
    # Text position 是在 ax data coords (0-1).
    # 所以 text 在 axes 中的 x 范围是 [target_left, max_text_right]
    # Colorbar 在 axes 中的 x 位置？不，Colorbar 是 Figure Axes.
    # 这导致了坐标系不统一。
    
    # 简化策略：
    # 1. 计算文本在 Data Coords 的垂直范围 [text_bottom, text_top]
    # 2. 调整 ax.ylim 贴合这个范围 (加上 padding)
    # 3. 调整 Figure Height 使得 1 unit data 对应原本的物理高度，从而保持纵横比
    
    pad_y = 0.05 # 上下留白 (Data Units)
    data_height = (text_top + pad_y) - (text_bottom - pad_y)
    if data_height < 0.1: data_height = 0.1 # 防止过小
    
    # 新的 limits
    new_ylim_bottom = text_bottom - pad_y
    new_ylim_top = text_top + pad_y
    
    # 新的 Figure Height
    # 原本 1.0 Data Height = fig_height Inches
    # 现在 data_height Data Height = ? Inches
    # 我们希望 Scale 不变，即 1 Data Unit 仍然对应 fig_height Inches
    new_fig_height = data_height * fig_height
    
    # 应用高度调整
    fig.set_size_inches(12, new_fig_height)
    ax.set_ylim(new_ylim_bottom, new_ylim_top)
    
    # --- Colorbar 重定位 ---
    # 由于调整了 Figure Height 和 YLim，原来的 cbar_bottom 计算失效
    # 我们需要重新计算 Colorbar 在新 Figure 中的位置
    # 但 Colorbar 是独立 Axis，我们可以直接指定它在 Figure 中的位置 [left, bottom, width, height]
    # 我们希望它在右侧，垂直居中
    
    # Colorbar 参数
    cb_w = 0.03 # 宽度 (Figure Coordinates)
    cb_h = 0.8  # 高度 (相对 Figure Height)
    cb_x = 0.92 # 右侧位置
    cb_y = (1 - cb_h) / 2 # 垂直居中
    
    cax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
    
    # 创建 ScalarMappable
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # 添加 Colorbar
    cb = fig.colorbar(sm, cax=cax, orientation='vertical')
    # No Label
    cb.ax.tick_params(labelsize=24) # Increased font size
    
    return fig

def add_colorbar_to_image(pil_img, cmap='jet', label='Attention'):
    """
    给 PIL 图片右侧添加一个 Colorbar
    Args:
        pil_img: PIL.Image
        cmap: Matplotlib colormap name (default 'jet' to match cv2.COLORMAP_JET)
        label: Colorbar label
    Returns:
        PIL.Image: Combined image
    """
    from PIL import Image
    
    # Create a figure just for the colorbar
    # Aspect ratio optimized for a vertical strip
    # Increased figsize significantly to ensure high DPI rendering
    # Increased width from 3 to 6 to accommodate large labels (120pt)
    fig = plt.figure(figsize=(6, 16)) 
    # Adjusted axis position: [left, bottom, width, height]
    # left=0.1, width=0.15 (relative to 6 inches) -> bar is ~0.9 inch wide
    # Remaining right space is ~75% of 6 inches = 4.5 inches for text
    ax = fig.add_axes([0.1, 0.05, 0.15, 0.9])
    
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cb = fig.colorbar(sm, cax=ax, orientation='vertical')
    # Increased labelsize to ensure readability when downscaled
    cb.ax.tick_params(labelsize=140) # Increased from 120 to 140
    cb.outline.set_visible(False)
    
    # Render
    canvas = FigureCanvasAgg(fig)
    # Force high DPI
    fig.set_dpi(300)
    canvas.draw()
    
    # Convert to PIL
    s, (width, height) = canvas.print_to_buffer()
    bar_img = Image.frombytes("RGBA", (width, height), s)
    
    # Resize bar_img to match pil_img height
    w_orig, h_orig = pil_img.size
    
    # Calculate target width for bar based on height ratio
    # We want the bar to be fully visible but not overwhelming
    # Let's say bar width is roughly 15% of original image width? 
    # Or just scale proportionally to height.
    
    scale_factor = h_orig / height
    new_bar_w = int(width * scale_factor)
    new_bar_h = h_orig
    
    bar_img_resized = bar_img.resize((new_bar_w, new_bar_h), Image.Resampling.LANCZOS)
    
    # Create composite image
    total_width = w_orig + new_bar_w
    new_img = Image.new("RGB", (total_width, h_orig), (255, 255, 255))
    new_img.paste(pil_img, (0, 0))
    # Paste bar (handle alpha)
    new_img.paste(bar_img_resized, (w_orig, 0), mask=bar_img_resized.split()[3])
    
    plt.close(fig)
    return new_img
