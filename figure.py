import plotnine as p9
import pandas as pd

tasks = ["num", "num+text", "easy entity", "hard entity", "overall"]
models = ["Llama-3-8b", "Llama-3-70b", "Yi-6b", "Yi-34b"]

num = [0.11222222222222222, 0.1, 0.38857142857142857, 0.46300448430493274]
num_text = [
    0.34858681022880217,
    0.5477792732166891,
    0.85383502170767,
    0.7260458839406208,
]
easy_entity = [
    0.5313568985176739,
    0.7388888888888889,
    0.7438882421420256,
    0.7987654320987654,
]
hard_entity = [0.39281705948372614, 0.61, 0.6736196319018405, 0.5923423423423423]
overall = [
    0.3447669305189094,
    0.4969503340110369,
    0.6537037037037037,
    0.6376463524467127,
]

tasks = [num, num_text, easy_entity, hard_entity, overall]

df = pd.DataFrame(tasks, columns=models)
df["task"] = ["num", "num+text", "easy entity", "hard entity", "overall"]

df = pd.melt(
    df, id_vars=["task"], value_vars=models, var_name="model", value_name="consistency"
)

p = (
    p9.ggplot(df, p9.aes(x="task", y="consistency", fill="model"))
    + p9.geom_bar(stat="identity", position="dodge")
    + p9.theme_minimal()
    + p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1))
)

# save as pdf
p.save("results.pdf")
