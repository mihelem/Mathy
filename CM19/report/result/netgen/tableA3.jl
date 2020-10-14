path = "CM19/report/result/netgen/"
for img in images
    mv(path*img, path*replace(img, "*"=>"_"))
end
images = [img for img in readdir(path) if endswith(img, ".png")]
hiters = sort!([img for img in images if startswith(img, "hiter_fixlim")])
itepst = sort!([img for img in images if startswith(img, "itepst_fixlim")])

title_hiters = "Table 5. Restarted Nesterov Momentum : number of stages against \$\\epsilon_{rel}\$"
title_itepst = "Table 6. Restarted Nesterov Momentum : number of iterations per stage against \$\\epsilon_{rel}\$"
caption_hiters = "Table 5"
caption_itepst = "Table 6"
label_hiters = "table:hiters"
label_itepst = "table:itepst"

function make_prelude(title)
    "\\documentclass{article}
    \\usepackage[a3paper, mag=1000, left=0.5cm, right=0.5cm, top=0.5cm, bottom=0.5cm, headsep=0.7cm, footskip=0cm]{geometry}
    \\usepackage{graphicx}
    \\graphicspath{ {./img/} }
    \\usepackage{caption}
    \\usepackage{longtable}
    \\usepackage{lscape}

    \\begin{document}

    \\pagenumbering{gobble}
    \\begin{landscape}
    \\begin{center}
    \\textbf{$title}
    \\begin{longtable}{| c | c | c | c |}
    \\hline"
end

function make_epilogue(caption, label)
    #"\\caption{$caption}
    "\\label{$label}
    \\end{longtable}
    \\end{center}
    \\end{landscape}

    \\end{document}"
end

function write_table(
    name,
    prelude,
    epilogue,
    imgs)

    open(name, "w") do io
        println(io, prelude)
        for i in 1:4:length(imgs)
            for j in 0:2
                println(io,
                    "\\includegraphics[height=0.22\\textheight]{$(imgs[i+j])} &")
            end
            println(io,
                "\\includegraphics[height=0.22\\textheight]{$(imgs[i+3])} \\\\
                \\hline")
        end
        println(io, epilogue)
    end
end

write_table(
    "table_hiters.tex",
    make_prelude(title_hiters),
    make_epilogue(caption_hiters, label_hiters),
    hiters)

write_table(
    "table_itepst.tex",
    make_prelude(title_itepst),
    make_epilogue(caption_itepst, label_itepst),
    itepst)
