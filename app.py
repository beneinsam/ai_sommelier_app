import streamlit as st
from sommelier import search_wine, recommand_wine, describe_dish_flavor

st.title("AI 소믈리에 추천, 음식 짝궁 와인")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_image = st.file_uploader("요리 이미지를 업로드하세요.", type=["jpg", "jpeg", "png"])
    user_prompt = st.text_input("프롬프트를 입력하세요.", "이 요리에 어울리는 와인을 추천해주세요.")

with col2:
    if uploaded_image:
        st.image(uploaded_image, caption="업로드된 요리 이미지", use_container_width=True)

with col1:
    if st.button("추천하기"):
        if not uploaded_image:
            st.warning("이미지를 업로드해주세요.")
        else:
            with st.spinner("1단계: 요리의 맛과 향을 분석하는 중..."):
                dish_flavor = describe_dish_flavor(uploaded_image.read(), "이 요리의 이름과 맛과 향과 같은 특징을 한 문장으로 설명해줘.")
                st.markdown(f"#### 🍜 요리의 맛과 향 분석 결과")
                st.info(dish_flavor)

            with st.spinner("2단계: 요리에 어울리는 와인 리뷰를 검색하는 중..."):
                wine_search_result = search_wine(dish_flavor)
                st.markdown("#### 🍷 와인 리뷰 검색 결과")
                st.text(wine_search_result['wine_reviews'])
                for idx, (review, score) in enumerate(zip(wine_search_result['wine_reviews'], wine_search_result['similarities'])):
                    st.write(f"**[{idx+1}] 유사도: {score:.2f}**")
                    st.text(review)

            with st.spinner("3단계: AI 소믈리에가 와인 페어링에 대한 추천글을 생성하는 중..."):
                wine_recommandation = recommand_wine({
                    "dish_flavor": dish_flavor,
                    "wine_reviews": wine_search_result['wine_reviews'],
                })
                st.markdown("#### 🍷 AI 소믈리에의 와인 추천")
                st.info(wine_recommandation)
            st.success("추천이 완료되었습니다!")




st.markdown(
    """
    <hr style="margin-top: 50px;">
    <div style="text-align: center; color: gray;">
        © 2025 와인 추천 앱 | Build by HumanAI Solution
    </div>
    """,
    unsafe_allow_html=True
)