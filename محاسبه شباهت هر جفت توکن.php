// محاسبه شباهت هر جفت توکن (dot product)
function dot($a, $b) {
    $s = 0;
    for ($i = 0; $i < count($a); $i++) $s += $a[$i] * $b[$i];
    return $s;
}

$n = count($embeddings);
$attention = [];
for ($i = 0; $i < $n; $i++) {
    $scores = [];
    for ($j = 0; $j < $n; $j++) {
        $scores[] = dot($embeddings[$i], $embeddings[$j]);
    }
    // نرمال‌سازی با softmax (شبیه sigmoid ولی برای چند گزینه)
    $max = max($scores);
    $exp = array_map(fn($s) => exp($s - $max), $scores);
    $sum = array_sum($exp);
    $attention[$i] = array_map(fn($e) => $e / $sum, $exp);
}

// نمایش: کدوم کلمه به کدوم توجه داره
echo "=== Attention (توجه) ===\n";
for ($i = 0; $i < $n; $i++) {
    echo "«{$tokens[$i]}» → ";
    for ($j = 0; $j < $n; $j++) {
        printf("%s:%.2f ", $tokens[$j], $attention[$i][$j]);
    }
    echo "\n";
}