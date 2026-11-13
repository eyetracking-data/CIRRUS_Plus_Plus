"""Shared component colors for quality cards and charts."""

from html import escape

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from cirruspp_core import quality_breakdown, quality_index

COMPONENT_COLORS = {
    "Completeness": "#4477AA",
    "Continuity": "#278579",
    "Outlier burden": "#B57A25",
    "Distribution shape": "#8870AD",
    "Temporal stability": "#BC6B63",
    "Cross-signal consistency": "#AF658B",
}


def component_rgba(component, alpha):
    color = COMPONENT_COLORS[component].lstrip("#")
    red, green, blue = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def quality_component_chart(profile, weights):
    breakdown=quality_breakdown(profile,weights)  # noqa: F405
    colors=[COMPONENT_COLORS[label] for label in breakdown.component]
    fig=go.Figure(go.Bar(x=breakdown.score_0_100,y=breakdown.component,orientation="h",
                        text=breakdown.score_0_100.map(lambda value:f"{value:.1f}"),
                        marker=dict(color=colors),width=.58,
                        customdata=breakdown[["domain_weight","weighted_points"]].to_numpy(),
                        hovertemplate="%{y}<br>Score: %{x:.1f}/100<br>Weight: %{customdata[0]:.0%}<br>Contribution: %{customdata[1]:.2f} points<extra></extra>"))
    fig.update_traces(textposition="outside",cliponaxis=False)
    fig.update_layout(showlegend=False,xaxis=dict(range=[0,112],tickvals=[0,25,50,75,100],
                      showgrid=True,gridcolor="rgba(128,145,170,.12)",zeroline=False),
                      yaxis=dict(autorange="reversed",showgrid=False,ticks=""))
    fig.update_layout(height=340,xaxis_title=None,yaxis_title=None,
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0,r=20,t=15,b=60),font=dict(size=12,color="#64748b"))
    fig.update_traces(marker_cornerradius=4)
    return fig

def render_quality_calculation(profile, weights, *, key):
    breakdown = quality_breakdown(profile, weights)
    total = quality_index(profile, weights)
    st.caption("Component score × domain weight = contribution. Add the six contributions to get the index.")
    st.markdown("""<style>
.quality-calculation {width:100%;border-collapse:collapse;font-size:.8rem;}
.quality-calculation th {font-weight:500;opacity:.65;font-size:.72rem;}
.quality-calculation th,.quality-calculation td {padding:9px 4px;text-align:right;border-bottom:1px solid rgba(128,145,170,.13);font-variant-numeric:tabular-nums;}
.quality-calculation th:first-child,.quality-calculation td:first-child {text-align:left;}
.quality-calculation-dot {display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;}
.quality-calculation-total {display:flex;justify-content:space-between;align-items:baseline;gap:12px;padding-top:12px;font-size:.85rem;}
.quality-calculation-total strong {font-size:1.45rem;letter-spacing:-.025em;}
</style>""", unsafe_allow_html=True)
    rows = []
    for row in breakdown.itertuples():
        color = COMPONENT_COLORS[row.component]
        rows.append(
            f'<tr><td><span class="quality-calculation-dot" style="background:{color}"></span>{escape(row.component)}</td>'
            f'<td>{row.score_0_100:.2f}</td><td>× {row.domain_weight:.1%}</td>'
            f'<td style="color:{color};font-weight:650">= {row.weighted_points:.2f}</td></tr>'
        )
    st.markdown(
        '<table class="quality-calculation"><thead><tr><th>Component</th><th>Score</th><th>Weight</th><th>Points</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table>'
        + f'<div class="quality-calculation-total"><span>Sum of contributions</span><strong>{total:.2f} / 100</strong></div>',
        unsafe_allow_html=True,
    )
    if np.isfinite(total):
        fig = go.Figure()
        for row in breakdown.itertuples():
            fig.add_trace(go.Bar(
                x=[row.weighted_points], y=["Quality index"], name=row.component,
                orientation="h", marker_color=COMPONENT_COLORS[row.component],
                customdata=[[row.score_0_100, row.domain_weight]],
                hovertemplate=f"{escape(row.component)}<br>" + "%{customdata[0]:.2f} × %{customdata[1]:.1%} = %{x:.2f} points<extra></extra>",
            ))
        fig.update_layout(
            barmode="stack", height=85, showlegend=False, bargap=.5,
            xaxis=dict(range=[0,100],tickvals=[0,25,50,75,100],showgrid=False,zeroline=False),
            yaxis=dict(visible=False), margin=dict(l=0,r=5,t=5,b=25),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig,width="stretch",key=key,config={"displayModeBar":False})
    st.caption("Colors identify components. Values shown are rounded; the total uses full precision.")
