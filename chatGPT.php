// لایه خروجی: برای هر کلمه ممکن، یه احتمال
$candidates = ["دوست‌دخترم", "ترک", "کرد", "باران", "غذا", "خواب"];
$W_out = [
    [0.8, 0.3],   // دوست‌دخترم
    [0.5, 0.7],   // ترک
    [0.2, 0.6],   // کرد
    [-0.3, 0.1],  // باران
    [-0.5, -0.2], // غذا
    [-0.1, -0.4], // خواب
];

$logits = [];
foreach ($W_out as $w) {
    $z = 0;
    for ($j = 0; $j < 2; $j++) $z += $w[$j] * $hidden[$j];
    $logits[] = $z;
}

// softmax → احتمال
$max = max($logits);
$exp = array_map(fn($l) => exp($l - $max), $logits);
$sum = array_sum($exp);
$probs = array_map(fn($e) => $e / $sum, $exp);

echo "\n=== پیش‌بینی کلمه بعدی ===\n";
foreach ($candidates as $i => $c) {
    printf("%-12s → %.1f%%\n", $c, $probs[$i] * 100);
}