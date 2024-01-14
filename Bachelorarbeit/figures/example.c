int getImpact()
{
    scanf("%d %d %d", &a, &b, &c);
    Impact = 0;
    Division = 1;
    Sum = a + b;
    if ((a > 0) && (b > 0))
        Division = a / b;
    Max = b;
    if (a > b)
        Max = b; // Correct: Max = a;
    if (c == 1)
        Impact = Sum;
    if (c == 2)
        Impact = Division;
    if (c == 3)
        Impact = Max;
    return Impact;
}