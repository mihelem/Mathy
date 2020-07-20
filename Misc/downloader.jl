module Downloader
using HTTP, Gumbo, AbstractTrees

verbosity=0

function get_links(
    exts::Union{Array{String, 1}, Nothing, String},
    page::String)

    if typeof(exts) == String
        exts = [exts]
    end
    HTTP.request("GET", page; verbose=verbosity) |>
        r -> get_links(exts, r)
end

function check_ext(ext, name)
    name[end-length(ext)+1:end] == ext
end

function get_links(
    exts::Union{Array{String, 1}, Nothing, String},
    r::HTTP.Messages.Response)

    if typeof(exts) == String
        exts = [exts]
    end

    links = String[]
    if r.status == 200
        doc = parsehtml(String(r.body))
        for node in PreOrderDFS(doc.root)
            if typeof(node) == HTMLElement{:a}
                if haskey(node.attributes, "href")
                    href = node.attributes["href"]
                    if exts === nothing || any(check_ext.(exts, href))
                        push!(links, href)
                    end
                end
            end
        end
    end
    links
end

function make_absolute_link(
    link::String,
    page::String)

    j = length(page)
    for i in length(page):-1:1
        if (page[i] == '/') && (i>1) && (page[i-1] != '/')
            j = i
            break
        end
    end
    prefix = page[1:j]
    if prefix[j] != '/'
        prefix = prefix * "/"
    end
    function add_prefix(l)
        if any(startswith.(l, ["http://", "https://", "ftp://", "ftps://", "/"]))
            return l
        end
        prefix*l
    end

    add_prefix(link)
end

function get_absolute_links(
    exts::Union{Array{String, 1}, Nothing, String},
    r::HTTP.Messages.Response,
    page::String)

    make_absolute_link.(
        get_links(exts, r),
        page)
end

function get_absolute_links(
    exts::Union{Array{String, 1}, Nothing, String},
    page::String)

    make_absolute_link.(
        get_links(exts, page),
        page)
end

function download_to(source::String, dest::String)
    HTTP.request("GET", source; verbose=verbosity) |>
    r -> begin
        if r.status == 200
            open(dest, "w") do f
                write(f, r.body)
            end
        end
    end
end

function download(source::String, prefix::String)
    prefix * "/" * split(source, "/")[end] |>
    dest -> download_to(source, dest)
end

function gead!(
    exts::Union{Array{String, 1}, Nothing, String},
    page::String,
    dest_folder::String)

    if length(dest_folder) == 0 || dest_folder[end] != '/'
        dest_folder = dest_folder*"/"
    end
    get_absolute_links(exts, page) |>
    l -> download.(l, dest_folder)
end

end

"""
# Usage
I want to download in the folder `/home/mihele/Documenti/cm/Boyd/EE364b` all the
course notes I find at `http://stanford.edu/class/ee364b/lectures.html`; so,
after including this file, I'll type:
```julia
Downloader.gead!(
    [".pdf"],
   "http://stanford.edu/class/ee364b/lectures.html",
   "/home/mihele/Documenti/cm/Boyd/EE364b")
```

The function will return the array of dimensions of the downloaded files.
"""
