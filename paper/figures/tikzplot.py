"""Shared TikZ scaffolding for the data figures (Nature style: Helvetica at print size,
boxed panels, colour-blind-safe accents). Each figure is one \\begingroup with its own prefix."""

COL = 8.6          # spconf column width (cm): (178 - 6) / 2


# ----------------------------------------------------------------- style ----
def preamble(pfx, width, y0, y1):
    """Shared Nature-style scaffolding, with every name carrying this figure's prefix."""
    return rf"""\begingroup
\renewcommand{{\sffamily}}{{\fontfamily{{phv}}\selectfont}}%
\renewcommand{{\bfseries}}{{\fontseries{{b}}\selectfont}}%
\def\{pfx}W{{{width}}}%
\newcommand{{\{pfx}L}}{{\fontsize{{8}}{{9.6}}\selectfont}}%
\newcommand{{\{pfx}M}}{{\fontsize{{7}}{{8.4}}\selectfont}}%
\newcommand{{\{pfx}S}}{{\fontsize{{6}}{{7.0}}\selectfont}}%
\definecolor{{{pfx}Ink}}{{RGB}}{{0,0,0}}
\definecolor{{{pfx}Line}}{{RGB}}{{75,75,75}}
\definecolor{{{pfx}Rule}}{{RGB}}{{185,185,185}}
\definecolor{{{pfx}Bg}}{{RGB}}{{243,244,246}}
\definecolor{{{pfx}Gray}}{{RGB}}{{170,170,170}}
\definecolor{{{pfx}A}}{{RGB}}{{0,114,178}}
\definecolor{{{pfx}B}}{{RGB}}{{230,159,0}}
\definecolor{{{pfx}Bad}}{{RGB}}{{213,94,0}}
\definecolor{{{pfx}OK}}{{RGB}}{{0,158,115}}
\definecolor{{{pfx}Flag}}{{RGB}}{{204,121,167}}
\newcommand{{\{pfx}head}}[4]{{%
  \node[anchor=base west, inner sep=0pt, font=\sffamily\bfseries\{pfx}L] at (#1,#2) {{#3}};
  \node[anchor=base west, inner sep=0pt, font=\sffamily\bfseries\{pfx}M] at (#1+0.33,#2) {{#4}};}}
\newcommand{{\{pfx}panel}}[4]{{%
  \draw[{pfx}Ink, line width=0.5pt] ([shift={{(0.25pt,0.25pt)}}]#1,#3) rectangle ([shift={{(-0.25pt,-0.25pt)}}]#2,#4);}}
\newcommand{{\{pfx}divide}}[3]{{\draw[{pfx}Ink, line width=0.5pt] ({{#1}},#2) -- ({{#1}},#3);}}
\begin{{tikzpicture}}[
  x=1cm, y=1cm, font=\sffamily\{pfx}M, text={pfx}Ink,
  >={{Latex[length=3pt,width=2.4pt]}},
  line/.style={{line width=0.5pt, draw={pfx}Line}},
  ax/.style={{line width=0.45pt, draw={pfx}Line}},
  lbl/.style={{inner sep=0pt, align=center}},
  bold/.style={{lbl, font=\sffamily\bfseries\{pfx}M}},
  note/.style={{inner sep=0pt, align=center, font=\sffamily\{pfx}S}},
  noteL/.style={{inner sep=0pt, align=left, font=\sffamily\{pfx}S}},
  noteR/.style={{inner sep=0pt, align=right, font=\sffamily\{pfx}S}},
  dot/.style={{circle, draw={pfx}Ink, line width=0.4pt, inner sep=0pt, minimum size=3.1pt}},
]
\path[use as bounding box] (0,{y0}) rectangle (\{pfx}W,{y1});
"""


def close():
    return "\\end{tikzpicture}%\n\\endgroup\n"


class Axes:
    """Maps data units to cm inside a panel body, and draws the axis furniture."""

    def __init__(self, x0, x1, y0, y1, xlo, xhi, ylo, yhi):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.xlo, self.xhi, self.ylo, self.yhi = xlo, xhi, ylo, yhi

    def X(self, v):
        return self.x0 + (v - self.xlo) / (self.xhi - self.xlo) * (self.x1 - self.x0)

    def Y(self, v):
        return self.y0 + (v - self.ylo) / (self.yhi - self.ylo) * (self.y1 - self.y0)

    def frame(self, pfx, xticks, yticks, xfmt="{:g}", yfmt="{:g}"):
        s = [f"\\draw[ax] ({self.x0:.3f},{self.y1:.3f}) -- ({self.x0:.3f},{self.y0:.3f})"
             f" -- ({self.x1:.3f},{self.y0:.3f});"]
        for t in xticks:
            s.append(f"\\draw[ax] ({self.X(t):.3f},{self.y0:.3f}) -- ++(0,-0.055);")
            s.append(f"\\node[note, below=1pt] at ({self.X(t):.3f},{self.y0 - 0.055:.3f})"
                     f" {{{xfmt.format(t)}}};")
        for t in yticks:
            s.append(f"\\draw[ax] ({self.x0:.3f},{self.Y(t):.3f}) -- ++(-0.055,0);")
            s.append(f"\\node[note, left=1pt] at ({self.x0 - 0.055:.3f},{self.Y(t):.3f})"
                     f" {{{yfmt.format(t)}}};")
        return "\n".join(s)


# ------------------------------------------------------------------ data ----
