import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

st.set_page_config(page_title="Web Mercantile Warehouse Lease Optimizer", page_icon="🏭", layout="wide")

def solve_warehouse_leasing(required_space, lease_costs):
    requirements = np.array(required_space, dtype=float)
    costs = np.array(lease_costs, dtype=float)
    n_months = len(requirements)

    variables = []
    objective = []

    for start in range(n_months):
        for duration in range(1, n_months - start + 1):
            variables.append((start, duration))
            objective.append(costs[duration - 1])

    A_ub = []
    b_ub = []

    for month in range(n_months):
        row = []
        for start, duration in variables:
            active = start <= month < start + duration
            row.append(-1.0 if active else 0.0)
        A_ub.append(row)
        b_ub.append(-requirements[month])

    result = linprog(
        c=np.array(objective),
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=[(0, None)] * len(variables),
        method="highs",
    )

    if not result.success:
        raise RuntimeError(result.message)

    leases = []
    coverage = np.zeros(n_months)

    for value, (start, duration) in zip(result.x, variables):
        if value > 1e-7:
            end = start + duration - 1
            rate = costs[duration - 1]
            lease_cost = value * rate

            leases.append({
                "Start Month": start + 1,
                "Duration (Months)": duration,
                "End Month": end + 1,
                "Sq. Ft. Leased": value,
                "Cost / Sq. Ft.": rate,
                "Lease Cost": lease_cost,
            })

            coverage[start:end + 1] += value

    leases.sort(key=lambda x: (x["Start Month"], x["Duration (Months)"]))

    return {
        "total_cost": result.fun,
        "leases": leases,
        "coverage": coverage.tolist(),
        "requirements": requirements.tolist(),
    }

st.title("🏭 Web Mercantile Warehouse Lease Optimizer")
st.write("This app finds the minimum-cost combination of warehouse leases that satisfies each month's space requirement.")

default_requirements = [30000, 20000, 40000, 10000, 50000]
default_costs = [65, 100, 135, 160, 190]

left, right = st.columns(2)

with left:
    st.subheader("Monthly Space Requirements")
    req_df = pd.DataFrame({
        "Month": [1, 2, 3, 4, 5],
        "Required Space (Sq. Ft.)": default_requirements,
    })
    req_df = st.data_editor(req_df, hide_index=True, disabled=["Month"], use_container_width=True)

with right:
    st.subheader("Lease Costs")
    cost_df = pd.DataFrame({
        "Leasing Period (Months)": [1, 2, 3, 4, 5],
        "Cost per Sq. Ft.": default_costs,
    })
    cost_df = st.data_editor(cost_df, hide_index=True, disabled=["Leasing Period (Months)"], use_container_width=True)

if st.button("Optimize Leasing Plan", type="primary", use_container_width=True):
    try:
        requirements = req_df["Required Space (Sq. Ft.)"].astype(float).tolist()
        lease_costs = cost_df["Cost per Sq. Ft."].astype(float).tolist()

        result = solve_warehouse_leasing(requirements, lease_costs)

        st.success("Optimal solution found.")
        st.metric("Minimum Total Leasing Cost", f"${result['total_cost']:,.0f}")

        st.subheader("Optimal Lease Plan")
        lease_df = pd.DataFrame(result["leases"])

        if not lease_df.empty:
            formatted = lease_df.copy()
            formatted["Sq. Ft. Leased"] = formatted["Sq. Ft. Leased"].map(lambda x: f"{x:,.0f}")
            formatted["Cost / Sq. Ft."] = formatted["Cost / Sq. Ft."].map(lambda x: f"${x:,.2f}")
            formatted["Lease Cost"] = formatted["Lease Cost"].map(lambda x: f"${x:,.0f}")
            st.dataframe(formatted, hide_index=True, use_container_width=True)

        st.subheader("Monthly Coverage")
        coverage_df = pd.DataFrame({
            "Month": range(1, len(result["requirements"]) + 1),
            "Required Space": result["requirements"],
            "Available Leased Space": result["coverage"],
        })
        coverage_df["Extra Space"] = coverage_df["Available Leased Space"] - coverage_df["Required Space"]

        st.dataframe(
            coverage_df.style.format({
                "Required Space": "{:,.0f}",
                "Available Leased Space": "{:,.0f}",
                "Extra Space": "{:,.0f}",
            }),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Required vs. Available Space")
        chart_df = coverage_df.set_index("Month")[["Required Space", "Available Leased Space"]]
        st.bar_chart(chart_df)

        st.subheader("Cost Calculation")
        for lease in result["leases"]:
            st.write(
                f"Month {lease['Start Month']}: "
                f"{lease['Sq. Ft. Leased']:,.0f} sq. ft. for "
                f"{lease['Duration (Months)']} month(s) × "
                f"${lease['Cost / Sq. Ft.']:,.0f} = "
                f"${lease['Lease Cost']:,.0f}"
            )

        st.markdown(f"## Total Minimum Cost: **${result['total_cost']:,.0f}**")

    except Exception as e:
        st.error(f"Optimization error: {e}")
else:
    st.info("The default values are the Web Mercantile problem. Click **Optimize Leasing Plan** to calculate the solution.")
