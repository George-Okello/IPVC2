
# Add this to the TOP of your streamlit_app.py file, right after the imports

# VISUALIZATION FIX - Safe chart creation
def safe_plotly_chart(chart_type, data, **kwargs):
    """Create plotly charts safely without AttributeError"""
    try:
        import plotly.express as px

        if chart_type == "bar":
            fig = px.bar(data, **kwargs)
        elif chart_type == "scatter":
            fig = px.scatter(data, **kwargs)
        elif chart_type == "pie":
            fig = px.pie(data, **kwargs)
        elif chart_type == "line":
            fig = px.line(data, **kwargs)
        else:
            st.error(f"Unsupported chart type: {chart_type}")
            return None

        # SAFE axis update - only if method exists
        if hasattr(fig, 'update_xaxis'):
            fig.update_xaxis(tickangle=45)

        return fig

    except Exception as e:
        st.error(f"Chart creation failed: {e}")
        st.dataframe(data)  # Show table instead
        return None

# THEN FIND AND REPLACE these functions in your streamlit_app.py:

# REPLACE the display_preprocessing_analysis function with this:
def display_preprocessing_analysis(preprocessing_analysis):
    """Fixed preprocessing analysis display"""
    if not preprocessing_analysis or 'error' in preprocessing_analysis:
        st.error("No valid preprocessing analysis available")
        return

    # Overall statistics
    st.subheader("📈 Preprocessing Overview")
    overall = preprocessing_analysis.get('overall_statistics', {})

    if overall:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Images", overall.get('total_images', 0))
        with col2:
            st.metric("✅ Standard Success", overall.get('images_solved_by_standard', 0))
        with col3:
            st.metric("🔄 Rotation Success", overall.get('images_solved_by_rotation', 0))
        with col4:
            st.metric("🔧 Preprocessing Required", overall.get('images_requiring_preprocessing', 0))

    # Method effectiveness with FIXED charts
    method_effectiveness = preprocessing_analysis.get('method_effectiveness', {})
    if method_effectiveness:
        st.subheader("🏆 Individual Method Effectiveness")

        method_df = pd.DataFrame([
            {
                'Method': method.replace('_', ' ').title(),
                'Images Attempted': stats['images_attempted'],
                'Images Successful': stats['images_successful'],
                'Success Rate (%)': stats['success_rate'],
                'Codes Found': stats['total_codes_found'],
                'Effectiveness Score': stats['effectiveness_score']
            }
            for method, stats in method_effectiveness.items()
        ])

        method_df = method_df.sort_values('Success Rate (%)', ascending=False)
        st.dataframe(method_df, use_container_width=True)

        # FIXED chart
        st.subheader("📊 Method Effectiveness Visualization")
        fig = safe_plotly_chart(
            "bar",
            method_df.head(10),
            x='Method',
            y='Success Rate (%)',
            color='Effectiveness Score',
            title='Top 10 Preprocessing Methods by Success Rate'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Category analysis with FIXED charts
    category_analysis = preprocessing_analysis.get('category_analysis', {})
    if category_analysis:
        st.subheader("📂 Category Analysis")

        category_df = pd.DataFrame([
            {
                'Category': category.replace('_', ' ').title(),
                'Success Rate (%)': stats['success_rate'],
                'Methods Count': stats['methods_count'],
                'Total Successes': stats['total_successes']
            }
            for category, stats in category_analysis.items()
        ])

        st.dataframe(category_df, use_container_width=True)

        # FIXED pie chart
        fig_pie = safe_plotly_chart(
            "pie",
            category_df,
            values='Total Successes',
            names='Category',
            title='Success Distribution by Preprocessing Category'
        )
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)

# FIND any other chart creation code and replace it like this:

# OLD CODE (causes error):
# fig = px.bar(df, x='column1', y='column2')
# fig.update_xaxis(tickangle=45)
# st.plotly_chart(fig)

# NEW CODE (safe):
# fig = safe_plotly_chart("bar", df, x='column1', y='column2')
# if fig:
#     st.plotly_chart(fig, use_container_width=True)
