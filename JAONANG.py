import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import re
import os
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

font_file = 'THSarabunNew.ttf'

if os.path.exists(font_file):
    fm.fontManager.addfont(font_file)
    prop = fm.FontProperties(fname=font_file)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['font.size'] = 14
    mpl.rc('font', family=prop.get_name())
else:
    plt.rcParams['font.family'] = 'Tahoma'
    mpl.rc('font', family='Tahoma', size=11)

@st.cache_data(show_spinner=False)
def load_data_transposed():
    file_name = 'ตาราง 59-67.xlsx'
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_name)
        else:
            df = pd.read_excel(file_name, engine='openpyxl')
    except Exception as e:
        return None, str(e)

    try:
        first_col = df.columns[0]
        df = df.dropna(subset=[first_col])
        df[first_col] = df[first_col].astype(str).str.strip()
        
        df.set_index(first_col, inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        df_t = df.T 
        
        new_index = []
        for idx in df_t.index:
            match = re.search(r'\d{4}', str(idx))
            if match:
                new_index.append(match.group(0))
            else:
                new_index.append(str(idx)) 
        
        df_t.index = new_index
        df_t.index.name = 'Year'

        df_t = df_t.apply(pd.to_numeric, errors='coerce').fillna(0)
        df_t['ยอดรวมทุกประเภท (Grand Total)'] = df_t.sum(axis=1)

        return df_t, None

    except Exception as e:
        return None, f"Error: {e}"

st.set_page_config(page_title="Accident Analytics", layout="wide")
st.title("📊 ระบบวิเคราะห์สถิติแบบเจาะลึก")

df_data, error_msg = load_data_transposed()

if error_msg:
    st.error(f"❌ Error: {error_msg}")

elif df_data is not None:

    st.sidebar.header("🔍 1. เลือกหมวดหมู่")
    
    categories = list(df_data.columns)
    target_col = 'ยอดรวมทุกประเภท (Grand Total)'
    
    if target_col in categories:
        categories.remove(target_col)
        categories.insert(0, target_col)

    selected_category = st.sidebar.radio("รายการ:", categories)

    st.sidebar.markdown("---")
    st.sidebar.header("📈 2. เลือกรูปแบบกราฟ")
    
    graph_type = st.sidebar.selectbox(
        "เครื่องมือวิเคราะห์:",
        ["Time Series (กราฟเส้นปกติ)", 
         "Trend Analysis (เส้นแนวโน้ม)", 
         "Moving Average (ค่าเฉลี่ยเคลื่อนที่)", 
         "Exponential Smoothing (ปรับเรียบ)",
         "Decomposition (แยกองค์ประกอบ)"]
    )

    st.subheader(f"ผลการวิเคราะห์: {selected_category}")
    series_data = df_data[selected_category]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📄 ตารางข้อมูล")
        st.dataframe(series_data, height=400, use_container_width=True)
        
        total = series_data.sum()
        avg = series_data.mean()
        max_v = series_data.max()
        
        st.info(f"""
        **สรุปภาพรวม:**
        - รวมทั้งหมด: {total:,.0f}
        - เฉลี่ยต่อปี: {avg:,.2f}
        - สูงสุด: {max_v:,.0f}
        """)

    with col2:
        st.markdown(f"### 📉 กราฟ: {graph_type}")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        main_color = '#c0392b' if selected_category == 'ยอดรวมทุกประเภท (Grand Total)' else '#2980b9'

        if graph_type == "Time Series (กราฟเส้นปกติ)":
            ax.plot(series_data.index, series_data.values, marker='o', linewidth=2, color=main_color, label='ข้อมูลจริง')
            ax.fill_between(series_data.index, series_data.values, color=main_color, alpha=0.1)
            for x, y in zip(series_data.index, series_data.values):
                ax.annotate(f'{y:,.0f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
            ax.legend()

        elif graph_type == "Trend Analysis (เส้นแนวโน้ม)":
            x_nums = np.arange(len(series_data))
            z = np.polyfit(x_nums, series_data.values, 1)
            p = np.poly1d(z)
            ax.plot(series_data.index, series_data.values, 'o', color=main_color, alpha=0.5, label='ข้อมูลจริง')
            ax.plot(series_data.index, p(x_nums), linestyle='--', color='#e67e22', linewidth=3, label='แนวโน้ม (Trend)')
            ax.legend()

        elif graph_type == "Moving Average (ค่าเฉลี่ยเคลื่อนที่)":
            window = st.sidebar.slider("Window Size", 2, 5, 3)
            ma = series_data.rolling(window=window).mean()
            ax.plot(series_data.index, series_data.values, marker='o', color=main_color, alpha=0.3, label='ข้อมูลจริง')
            ax.plot(series_data.index, ma, color='#16a085', linewidth=3, label=f'Moving Avg ({window} ปี)')
            ax.legend()

        elif graph_type == "Exponential Smoothing (ปรับเรียบ)":
            alpha = st.sidebar.slider("Alpha", 0.1, 1.0, 0.3)
            try:
                model = SimpleExpSmoothing(series_data.values).fit(smoothing_level=alpha, optimized=False)
                ax.plot(series_data.index, series_data.values, marker='o', color=main_color, alpha=0.3)
                ax.plot(series_data.index, model.fittedvalues, color='#8e44ad', linestyle='--', linewidth=3, label=f'Smoothing')
                ax.legend()
            except:
                st.warning("ข้อมูลน้อยเกินไป")

        elif graph_type == "Decomposition (แยกองค์ประกอบ)":
            if len(series_data) >= 6:
                try:
                    dummy_dates = pd.date_range(start='2016', periods=len(series_data), freq='Y')
                    ts_data = pd.Series(series_data.values, index=dummy_dates)
                    res = seasonal_decompose(ts_data, model='additive', period=2)
                    
                    fig.clf()
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                    ax1.plot(series_data.index, res.observed, color=main_color)
                    ax1.set_title('Observed')
                    ax2.plot(series_data.index, res.trend, color='#e67e22')
                    ax2.set_title('Trend')
                    ax3.plot(series_data.index, res.resid, color='#7f8c8d', linestyle=':')
                    ax3.set_title('Residual')
                except:
                    st.warning("Decomposition Error")
            else:
                st.warning("ข้อมูลต้องมี 6 ปีขึ้นไป")

        if graph_type != "Decomposition (แยกองค์ประกอบ)":
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlabel("ปี พ.ศ.")
        
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📊 เปรียบเทียบสัดส่วน (ปีล่าสุด)")
    
    latest_year = df_data.index[-1]
    df_compare = df_data.loc[latest_year].drop('ยอดรวมทุกประเภท (Grand Total)', errors='ignore')
    
    df_compare.index = df_compare.index.astype(str)
    df_compare = df_compare[df_compare.index != 'nan']
    df_compare = df_compare[~df_compare.index.str.contains('รวม|Total', case=False, na=False)]
    df_compare = df_compare.sort_values(ascending=False)

    if not df_compare.empty:
        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
        bars = ax_bar.bar(df_compare.index, df_compare.values, color='#34495e')
        plt.xticks(rotation=45, ha='right')
        ax_bar.set_title(f"จำนวนอุบัติเหตุแยกตามประเภท ปี {latest_year}")
        
        for bar in bars:
            height = bar.get_height()
            ax_bar.annotate(f'{height:,.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=11)

        st.pyplot(fig_bar)
