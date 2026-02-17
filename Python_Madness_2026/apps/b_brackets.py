import streamlit as st
import polars as pl

# Title
st.markdown('All Tournament Brackets Since 2008')

# Read CSV with Polars
AG = pl.read_csv('Python_Madness_2026/data/step05c_FUHistory.csv')

# Convert columns to numeric
AG = AG.with_columns([
    pl.col("AFSeed").cast(pl.Int32, strict=False),
    pl.col("AFScore").cast(pl.Int32, strict=False),
    pl.col("AUSeed").cast(pl.Int32, strict=False),
    pl.col("AUScore").cast(pl.Int32, strict=False)
])

# Year slider
p_year = st.slider('Year: ', 2008, 2025)

if p_year == 2020:
    st.markdown("No Bracket in 2020")

if p_year != 2020:
    st.markdown(f'Bracket for {p_year}')

    # Filter for year and games >= 1
    AGP = AG.filter((pl.col("Year") == p_year) & (pl.col("Game") >= 1))

    # Separate Favorite and Underdog views
    AGPF = AGP.select(["Year","Game","AFSeed","AFTeam","AFScore"])
    AGPU = AGP.select(["Year","Game","AUSeed","AUTeam","AUScore"])

    # Build STS string
    AGPF = AGPF.with_columns([
        (pl.col("AFSeed").cast(pl.Utf8) + " " + pl.col("AFTeam") + " " + pl.col("AFScore").cast(pl.Utf8)).alias("STS"),
        pl.lit(0).alias("ti")   # Favorite first
    ]).select(["Game","ti","STS"])
    
    AGPU = AGPU.with_columns([
        (pl.col("AUSeed").cast(pl.Utf8) + " " + pl.col("AUTeam") + " " + pl.col("AUScore").cast(pl.Utf8)).alias("STS"),
        pl.lit(1).alias("ti")   # Underdog second
    ]).select(["Game","ti","STS"])



    # Combine and sort
    AGC = pl.concat([AGPF, AGPU]).sort(["Game","ti"]).to_pandas().reset_index(drop=True)

    # Bracket headers
    br_headers = [
        'Round 1 W','Round 2 W','Round 3 W','Round 4 W','Final Four W','Champ',
        'Final Four E','Round 4 E','Round 3 E','Round 2 E','Round 1 E'
    ]
    df = pl.DataFrame({h: [""]*32 for h in br_headers}).to_pandas()

    # Fill West side
    for y in range(32):
        df.loc[y,'Round 1 W'] = AGC.loc[y,'STS']
    for y in range(0,31,2):
        df.loc[y,'Round 2 W'] = AGC.loc[int(y/2+64),'STS']
    for y in range(0,31,4):
        df.loc[y+1,'Round 3 W'] = AGC.loc[int(y/4+96),'STS']
    for y in range(0,31,8):
        df.loc[y+3,'Round 4 W'] = AGC.loc[int(y/8+112),'STS']
    for y in range(0,31,16):
        df.loc[y+7,'Final Four W'] = AGC.loc[int(y/16+120),'STS']

    # Fill East side
    for y in range(32):
        df.loc[y,'Round 1 E'] = AGC.loc[y+32,'STS']
    for y in range(0,31,2):
        df.loc[y,'Round 2 E'] = AGC.loc[int(y/2+80),'STS']
    for y in range(0,31,4):
        df.loc[y+1,'Round 3 E'] = AGC.loc[int(y/4+104),'STS']
    for y in range(0,31,8):
        df.loc[y+3,'Round 4 E'] = AGC.loc[int(y/8+116),'STS']
    for y in range(0,31,16):
        df.loc[y+7,'Final Four E'] = AGC.loc[int(y/16+122),'STS']

    # Championship
    df.loc[15,'Champ'] = AGC.loc[124,'STS']
    df.loc[16,'Champ'] = AGC.loc[125,'STS']

    Champ = AGP.to_pandas().iloc[-1]
    winner = Champ['AFTeam'] if Champ['AFScore'] > Champ['AUScore'] else Champ['AUTeam']
    df.loc[0,'Champ'] = winner

    # Display in Streamlit
    st.checkbox("Use container width", value=True, key="use_container_width")
    st.dataframe(df, height=1200, width="stretch" if st.session_state.use_container_width else "content")
