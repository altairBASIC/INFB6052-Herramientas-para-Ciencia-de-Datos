% File: src/formato_utem.cls
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{src/formato_utem}[2025/04/15 Clase personalizada UTEM - INFB6052]

% ----------------------------------------------------------------
% Cargar clase base
\LoadClass[final]{article}

% ----------------------------------------------------------------
% Paquetes requeridos
\RequirePackage{lmodern}
\RequirePackage{graphicx}
\RequirePackage{array}
\RequirePackage{fancyhdr}
\RequirePackage[top=0.4cm, bottom=2cm, left=3cm, right=3cm, headheight=28pt, includehead, letterpaper, headsep=1.2cm, footskip=1.2cm]{geometry}
\RequirePackage{background}
\RequirePackage{titlesec}
\RequirePackage{booktabs}
\RequirePackage{xcolor}
\RequirePackage{hyperref}
\RequirePackage{multirow}
\RequirePackage{gensymb}
\RequirePackage{enumitem}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{tikz}
\RequirePackage{titlesec, booktabs, hyperref, multirow, gensymb}
\RequirePackage{enumitem, amsmath, amssymb, background, float}
\RequirePackage[spanish,es-noshorthands]{babel}
\RequirePackage{longtable}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}

% ----------------------------------------------------------------
% Evitar cortes de palabras por guion en Español
% Valores altos hacen que LaTeX no rompa palabras con guion
\hyphenpenalty=10000
\exhyphenpenalty=10000

% Ajuste recomendado para fancyhdr: evita advertencia sobre \headheight
\setlength{\headheight}{40pt}

% ----------------------------------------------------------------
% Colores personalizables
\definecolor{blanco}{RGB}{255,255,255}
\definecolor{verde}{RGB}{0,128,0}
\definecolor{azulmarino}{RGB}{0,0,128}
\definecolor{gris}{RGB}{128,128,128}
\definecolor{naranjo}{RGB}{255,165,0}
\definecolor{verdeagua}{RGB}{0,128,128}
\definecolor{grisclaro}{RGB}{240,240,240}

% ----------------------------------------------------------------
% Comandos personalizables
\newcommand{\logo}{\includegraphics[width=3.5cm]{src/utem.jpg}}
\newcommand{\universidad}{Universidad Técnica Metropolitana}
\newcommand{\carrera}{Ingeniería Civil en Ciencia de Datos}
\newcommand{\codcurso}{INFB6052}
\newcommand{\nombrecurso}{Herramientas para Cs. de Datos}
\newcommand{\fecha}{Octubre de 2025}
\newcommand{\autor}{{
	Ignacio Ramírez \\
	Cristian Vergara \\
	Antonia Montecinos \\
	\textbf{Grupo 2}
}}

\newcommand{\setuniversidad}[1]{\renewcommand{\universidad}{#1}}
\newcommand{\setcarrera}[1]{\renewcommand{\carrera}{#1}}
\newcommand{\setcodcurso}[1]{\renewcommand{\codcurso}{#1}}
\newcommand{\setnombrecurso}[1]{\renewcommand{\nombrecurso}{#1}}
\newcommand{\setfecha}[1]{\renewcommand{\fecha}{#1}}
\newcommand{\setautor}[1]{\renewcommand{\autor}{#1}}
\newcommand{\setencactividad}[1]{\renewcommand{\encactividad}{#1}}
\newcommand{\setdescactividad}[1]{\renewcommand{\descactividad}{#1}}

\newcommand{\tituloazul}[2][1.5ex]{%
  \section*{%
    \noindent
    \colorbox[RGB]{0,32,96}{\parbox{\dimexpr\linewidth-2\fboxsep}{%
      \color{white}\strut \textbf{#2}%
    }}%
  }%
  \vspace*{#1}%
}

\setlength{\parindent}{0pt} % Elimina la sangría al inicio de cada párrafo
\setlength{\hangindent}{0pt} % Elimina la sangría francesa


% ----------------------------------------------------------------
% Formato de secciones
\titleformat{\section}
  {\normalfont\normalsize\bfseries}{\thesection.}{1em}{}

% ----------------------------------------------------------------
% Encabezado y pie de página
\fancyhf{}
\fancyheadoffset[L]{1cm}

\fancyhead[L]{
\begin{tikzpicture}[remember picture,overlay]

% Define el color de la línea superior
\fill[azulmarino] (-2, 0.1) rectangle (\paperwidth-8.1cm, 1cm); % Borde superior azul
\fill[verde] (\paperwidth+3cm, 0.1) rectangle (13.5, 1cm); % Borde superior verde

% Define el color de la línea inferior
\fill[verde] (-2,-\paperheight) rectangle (\paperwidth,-\paperheight+0.7cm); % Borde inferior verde
\fill[azulmarino] (\paperwidth-17.5cm,-\paperheight) rectangle (\paperwidth,-\paperheight+0.7cm); % Borde inferior azul
\end{tikzpicture}

% Define el logo de la universidad
\begin{tikzpicture}[remember picture, overlay]
    \node at (15.5, -0.5) {\logo}; % Cambia las coordenadas según sea necesario
\end{tikzpicture}

% Define el encabezado
\rmfamily \fontsize{10}{12}\selectfont
{\textcolor{verde}{\MakeUppercase{\codcurso}} - 
\textcolor{azulmarino}{\MakeUppercase{\nombrecurso}} \\ 
\textcolor{gris}{\MakeUppercase{\carrera}}}
}

\fancyfoot[C]{\fontsize{6pt}{6pt}\selectfont \color{gris} 
\hspace{2cm} \MakeUppercase{\universidad}  \ - \MakeUppercase{\autor} \hfill  \thepage}

% Define el pie de página
\fancyfoot[C]{
    \fontsize{6pt}{6pt}\selectfont \color{gris} 
    \hspace{2cm} \MakeUppercase{\universidad}  \hfill  \thepage
    }

\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0.05pt}

% ----------------------------------------------------------------
% Portada personalizada
\newcommand{\makemytitle}{%


\begin{titlepage}
  \begin{tikzpicture}[remember picture, overlay]
    \node at (0, 0) {\includegraphics[width=10cm]{src/utem.jpg}};
  \end{tikzpicture}
  \centering
  \vspace*{2cm}

  \rmfamily \fontsize{18}{12}\selectfont
  {\textcolor{blanco}{\MakeUppercase{\codcurso}}  
    \textcolor{blanco}{\MakeUppercase{\nombrecurso}} \\
    \textcolor{blanco}{\MakeUppercase{\carrera}}}

  \vspace{2cm}
  
  \rmfamily \fontsize{18}{12}\selectfont
  {\textcolor{verde}{\MakeUppercase{\codcurso}} - 
    \textcolor{azulmarino}{\MakeUppercase{\nombrecurso}} \\
    \textcolor{gris}{\MakeUppercase{\carrera}}}
  \vspace{5cm}

  {\LARGE\bfseries \encactividad}
  \vspace{1cm}
  {\LARGE\bfseries \textit{\descactividad}}
  \vspace{5cm}

  {\large \autor \par}
  \vspace{0.5cm}
  {\large \fecha \par}
\end{titlepage}
}

% ----------------------------------------------------------------
% Variables de actividad
\newcommand{\encactividad}{Proyecto Integrador Semestral: \par}
\newcommand{\descactividad}{Desarrollo de un sistema de análisis de datos utilizando herramientas de Big Data, analítica web, y servicios de transformación de datos.}
