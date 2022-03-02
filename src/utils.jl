struct NotImplementedError <: Exception 
    msg::String
end

function Base.showerror(io::IO, err::NotImplementedError)
    print(io, "NotImplementedError: ")
    print(io, err.msg)
end
