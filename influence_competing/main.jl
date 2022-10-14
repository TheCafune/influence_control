include("graph.jl")
include("core.jl")
include("edgecore.jl")
include("compare.jl")

using LinearAlgebra
using Laplacians

for allll=1:1

fname = open("filename.txt", "r")
str   = readline(fname);
nn     = parse(Int, str);

for nnnn=1:nn
str = readline(fname);
str = split(str);
G   = get_graph(str[1]);
on=G.n;om=G.m;
Gc=findconnect(G);
G=Gc;
n=G.n;m=G.m;

#######
kmax=100;
#######
fout=open("zlw_mid.txt", "a")
println(fout,str[1],' ',G.n,' ',G.m,' ',kmax);
# alpha=powerLaw(n)
# beta=powerLaw(n)
# alpha=Uniform(n)
# beta=Uniform(n)
alpha=Exponential(n)
beta=Exponential(n)
# Uniform
# Exponential

S=Int[];
T=Int[];
# t=Int(round(n/100))
t=5 ###
T=rand(1:n,t)
union!(T);
T=union(1:5);

L=lapsp(G);
LF=lapsp(G);
for i in T
    LF[i,i]+=beta[i]
end

t1=time()
ans1=fast(G,LF,n,alpha,beta,T,kmax)
t2=time()
ans2=exact(G,LF,n,alpha,beta,T,kmax)
t3=time()

# println(111)
# for kk=1:kmax
#     ans3=opt(G,L,n,alpha,beta,T,kk)
#     println(fout,ans3)
# end


ans4=[]
deg=zeros(n)
for i=1:n
    deg[i]=L[i,i]
end
for i=1:kmax
    ss=argmax(deg)
    while ss in ans4 || ss in T
        deg[ss]=0;
        ss=argmax(deg)
    end
    push!(ans4,ss)
end

ans5=[]
page=PageRankCentrality(G)
for i=1:kmax
    ss=argmax(page)
    while ss in ans5 || ss in T
        page[ss]=0;
        ss=argmax(page)
    end
    push!(ans5,ss)
end

ans6=[];
guzhi=zeros(n);
guzhi.=alpha;
for i=1:kmax
    ss=argmax(guzhi)
    while ss in ans6 || ss in T
        guzhi[ss]=0;
        ss=argmax(guzhi)
    end
    push!(ans6,ss)
end

ans7=[]
for i=1:kmax
    ss=rand(1:n)
    while ss in ans7 || ss in T
        ss=rand(1:n)
    end
    push!(ans7,ss)
end


ans8=[]
c=zeros(n)
for i in T
    c[i]=beta[i]
end
f=approxchol_sddm(LF)
kexi=f(c);
for i=1:kmax
    ss=argmax(kexi)
    while ss in ans8 || ss in T
        kexi[ss]=0
        ss=argmax(kexi)
    end
    push!(ans8,ss)
end

L=lapsp(G)
println(fout,"time exact fast=",t3-t2,' ',t2-t1)
println(fout,calc(L,n,alpha,beta,[],T))

for i=1:kmax
    # println(calc(ans1[1:i]),' ',calc(ans2[1:i]),' ',calc(ans3[1:i]),' ',calc(ans4[1:i]),' ',calc(ans5[1:i]))
    println(fout,calc(L,n,alpha,beta,ans1[1:i],T),' ',calc(L,n,alpha,beta,ans2[1:i],T),' ',calc(L,n,alpha,beta,ans4[1:i],T),' ',calc(L,n,alpha,beta,ans5[1:i],T),' ',calc(L,n,alpha,beta,ans6[1:i],T),' ',calc(L,n,alpha,beta,ans7[1:i],T),' ',calc(L,n,alpha,beta,ans8[1:i],T))
end
println(fout)
close(fout)

end
close(fname)
end
