function en = energy(psiL,psiR,H)

    en = scon({psiL,psiR,H,conj(psiL),conj(psiR)},...
    {[1 -1],[2 -2],[1 2 3 4],[3 -3],[4 -4]});

    en = real(en);

end
