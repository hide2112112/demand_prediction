from tokenize import PseudoExtras
from pyparsing import line
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import prophet.plot as fp
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
import altair as alt
import itertools

# ページの既定の設定を構成する
st.set_page_config(page_title="売上予測",
                   page_icon="👩‍💻",
                   initial_sidebar_state="collapsed",
                   )

# 関数の実行をメモするための関数デコレータ
@st.cache(persist=False,
          allow_output_mutation=True,
          suppress_st_warning=True,
          show_spinner= True)

def load_csv():

    # データを読み込む
    df_input = pd.DataFrame()

    # データを読み込む(CSV)
    df_input = pd.read_csv(input,sep=None ,
                            engine='python', #日本語や全角が入ったファイル読み込み時のエラー解決策
                            encoding='utf-8',
                            parse_dates=True, #True：indexの解析（indexが日付のときのみ機能します。）
                            infer_datetime_format=True #Trueのときparse_datesも有効なら処理速度が向上する可能性がある。
                            )    

    # データを返す
    return df_input

def prep_data(df):

    df_input = df.rename({date_col:"ds",metric_col:"y"},errors='raise',axis=1)
    st.markdown("選択された日付の列は**ds**、値の列は**y**と表示されるようになりました。")
    df_input = df_input[['ds','y']]
    df_input = df_input.sort_values(by='ds',ascending=True)
    return df_input

st.title("売上予測")
st.write("このアプリは、売上予測を行います。")
st.markdown("""使用する予測ライブラリーは、**[Prophet](https://facebook.github.io/prophet/)**です。""")
df = pd.DataFrame()

st.subheader('1.データの読み込み')
st.write("時系列データのcsvファイルをインポートする。")

input = st.file_uploader("Upload CSV", type=".csv")
example_file = ""

use_example_file = st.checkbox(
    "CSVファイル例", False, help="見本ファイルを表示します。"
)

if use_example_file:
    example_file = "sumple_u.csv"

if example_file:
    example_df = pd.read_csv(example_file)
    st.markdown("### 見本ファイル")
    st.dataframe(example_df.head())
    st.write("datetime列は、日付を表す列です。y列は、売上などの目的変数（数字）を表す列です。y列にある<NA>は欠損値を表し、CSVファイル上では空欄を意味します。")

if input:
    with st.spinner('データ読み込み中...'):
        df = load_csv()

        st.write("Columns:")
        st.write(list(df.columns))
        columns = list(df.columns)

        col1,col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("日付列の選択",index=0, options=columns,key="date")
        with col2:
            metric_col = st.selectbox("値列の選択",index=1, options=columns,key="values")
    
        df = prep_data(df)
        output = 0


if st.checkbox("チャートデータ",key="show"):
    if input:
        with st.spinner("データをプロットする………"):
            col1,col2 = st.columns(2)
            with col1:
                st.dataframe(df)

            with col2:
                st.write("データフレームの説明")
                st.write(df.describe())
        
        try:
            line_chart = alt.Chart(df).mark_line().encode(
                x='ds:T',
                y='y:Q',tooltip=['ds:T','y']).properties(title="時系列プレビュー").interactive()
            st.altair_chart(line_chart,use_container_width=True)

        except:
            st.line_chart(df['y'],use_container_width=True,height=300)

    if not input:
        st.warning('CSVファイルをアップロードする必要があります。')

st.subheader('2.パラメータ設定')

with st.container(): #複数要素コンテナを挿入します。
    st.write('このセクションでは、アルゴリズムの設定を変更することができます。')

    with st.expander("予測日数"):
        periods_input = st.number_input("予測する未来の期間（日数）を選択します。",
        min_value=1,max_value=366,value=90)

    with st.expander("季節性"):
        st.markdown("""デフォルトの季節性は加算型ですが、最適な選択は特定のケースに依存するため、特定のドメインの知識が必要です。より詳しい情報は、[ドキュメント](https://facebook.github.io/prophet/docs/multiplicative_seasonality.html)をご覧ください。""")
        seasonality = st.radio(label='季節性',options=['additive','multiplicative'])

    with st.expander("成長モデル"):
        st.write('Prophetはデフォルトで線形成長モデルを使用しています。')
        st.markdown("""詳しくは[ドキュメント](https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth)をご覧ください。""")

        growth = st.radio(label='成長モデル',options=['linear','logistic'])

        if growth == 'linear':
            growth_settings= {
                        'cap':1,
                        'floor':0
                    }
            cap=1
            floor=1
            df['cap']=1
            df['floor']=0

        if growth == 'logistic':
            st.info('サチュレーション設定')

            cap = st.slider('Cap',min_value=0.0,max_value=1.0,step=0.05)
            floor = st.slider('Floor',min_value=0.0,max_value=1.0,step=0.05)
            if floor > cap:
                st.error('無効な設定です。CapはFloorより高くなければなりません。')
                growth_settings={}
                
            if floor == cap:
                st.warning('CapはFloorより高くなければならない')
            else:
                growth_settings = {
                        'cap':cap,
                        'floor':floor
                        }
                df['cap']=cap
                df['floor']=floor

    with st.expander('ハイパーパラメーター'):
        st.write('このセクションでは、スケーリング係数を調整することが可能です。')

        seasonality_scale_values= [0.1, 1.0,5.0,10.0]    
        changepoint_scale_values= [0.01, 0.1, 0.5,1.0]

        st.write("チェンジポイントの事前スケールは、トレンドの柔軟性、特にトレンドのチェンジポイントでトレンドがどの程度変化するかを決定します。")
        changepoint_scale = st.select_slider(label = 'チェンジポイント先行スケール',options=changepoint_scale_values)

        st.write("シーズナリティ変更ポイントは、シーズナリティの柔軟性をコントロールします。")
        seasonality_scale = st.select_slider(label = '季節性先行型',options=seasonality_scale_values)

        st.markdown("""詳しくは[ドキュメント](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)をお読みください""")

with st.container():
    st.subheader("3.予測")

    if input:

        if st.checkbox("モデルの初期化",key="fit"):
            m = Prophet(seasonality_mode=seasonality,
                        growth=growth,
                        changepoint_prior_scale=changepoint_scale,
                        seasonality_prior_scale= seasonality_scale)
            m.add_country_holidays(country_name="JP")#日本の祝日
            m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            
            with st.spinner('モデルに当てはめる…'):

                m = m.fit(df)
                future = m.make_future_dataframe(periods=int(periods_input),freq='D')
                future['cap']=cap
                future['floor']=floor
                st.write("まで予測することができます。",future['ds'].max())
                st.success('モデルの適合に成功')
    
        if st.checkbox("予測の生成",key="predict"):
            # try:
                with st.spinner("予測しています…"):

                    forecast = m.predict(future)
                    st.success('予測値の生成に成功')
                    st.dataframe(forecast)
                    fig1 = m.plot(forecast)
                    st.write(fig1)
                    output = 1

                    if growth == 'linear':
                        fig2 = m.plot(forecast)
                        a = add_changepoints_to_plot(fig2.gca(), m, forecast)
                        st.write(fig2)
                        output = 1
                
                export_forecast = pd.DataFrame(forecast[['ds','yhat_lower','yhat','yhat_upper']])
                export_forecast_csv = export_forecast.to_csv(index=False)
            # except:
                # st.warning('まず、モデルをトレーニングする必要があります。')
    
        if st.checkbox('コンポーネントを表示する'):
            try:
                with st.spinner("読み込み中..."):
                    fig3 = m.plot_components(forecast)
                    st.write(fig3)
            except:
                st.warning("予測作成が必要です。")

st.subheader('4.モデル検証')
st.write("このセクションでは、モデルのクロスバリデーションを行うことで、モデルの性能を検証します。")
with st.expander("モデルの検証"):
    st.write(f"ここでは、はじめの365日分の学習データから始めて、30日ごとに予測を行い、30日間の予測性能を評価するクロスバリデーションを行う。")

    if input:
        if output == 1:
            metrics = 0
        if st.checkbox("指標の算出"):
            with st.spinner("実施中…"):
                try:
                    df_cv = cross_validation(m, initial='365 days',
                                            period='30 days',
                                            horizon = '30 days',
                                            parallel="processes")
                                


                    df_p= performance_metrics(df_cv)
                    st.write(df_p)
                    metrics = 1
                    fig4 = plot_cross_validation_metric(df_cv, metric="mae")
                    st.write(fig4)
                except:
                    st.warning('モデルをトレーニングする必要があります。')
                    metrics = 0

    else:
        st.write("指標を確認するための予測を作成する")

st.subheader("5. ハイパーパラメータの調整")
st.write("ここでは、ハイパーパラメータの最適な組み合わせを見つけることが可能である。")
st.markdown("""詳しくは、[ドキュメント](https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning)をご覧ください。""")

param_grid = {

                'changepoint_prior_scale': [0.01, 0.1, 0.5, 1.0],
                'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0],
            }

# パラメータの全組み合わせを生成する
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmse = [] # 各パラメータのRMSEをここに格納する

if input:
    if output == 1:

        if st.button("ハイパーパラメータを最適化する"):

            with st.spinner("最適な組み合わせを探す お待ちください。"):


                try:
                # クロスバリデーションで全パラメータを評価する
                    for params in all_params:
                        m = Prophet(**params).fit(df)
                        df_cv = cross_validation(m, initial='365 days',
                                                    period='30 days',
                                                    horizon = '30 days',
                                                    parallel="processes")
                        df_p = performance_metrics(df_cv, rolling_window=1)
                        rmse.append(df_p['rmse'].values[0])
                except:
                    for params in all_params:
                        m = Prophet(**params).fit(df) # 与えられたパラメータでモデルをフィットさせる
                        df_cv = cross_validation(m, initial='365 days',
                                                    period='30 days',
                                                    horizon = '30 days',
                                                    parallel="processes")
                        df_p = performance_metrics(df_cv, rolling_window=1)
                        rmse.append(df_p['rmse'].values[0])

            # 最適なパラメータを探す
            tuning_results = pd.DataFrame(all_params)
            tuning_results['rmse'] = rmse
            st.write(tuning_results)

            best_params = all_params[np.argmin(rmse)]

            st.write("最適なパラメータの組み合わせは")
            st.write(best_params)
            #st.write(f"Changepoint prior scale:  {best_params[0]} ")
            #st.write(f"Seasonality prior scale: {best_params[1]}  ")
            st.write(" これらのパラメータを使用して、コンフィギュレーションセクション2の処理を繰り返すことができます。")


    else:
        st.write("最適化するためのモデルを作る")

st.subheader('6.エクスポート')
st.write("CSVファイルをエクスポートします。")
if input:
    if output == 1:
        st.download_button(
            label="ダウンロード",
            data=export_forecast_csv,
            file_name='forecast.csv',
            mime='text/csv'
        )

    else:
        st.write("ダウンロードする予測を生成します。")




























