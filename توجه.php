<?php
// ============================================
// شبیه‌سازی ساده پیش‌بینی کلمه بعدی (مثل GPT)
// ============================================

// ===== ۱) داده‌های ورودی =====
$sentence = "من خیلی ناراحتم چون";
$tokens = explode(" ", $sentence);
// tokens = ["من", "خیلی", "ناراحتم", "چون"]

// ===== ۲) واژه‌نامه و Embedding =====
// در واقعیت ۱۲۲۸۸ بعده، اینجا ۳ بعدی برای سادگی
$vocab = [
    "من"         => [0.2, 0.8, 0.1],
    "خیلی"       => [0.5, 0.3, 0.9],
    "ناراحتم"    => [0.9, 0.1, 0.3],
    "چون"        => [0.1, 0.4, 0.7],
    "دوست‌دخترم" => [0.8, 0.2, 0.4],
    "ترک"        => [0.7, 0.5, 0.2],
    "کرد"        => [0.3, 0.6, 0.5],
    "باران"      => [0.1, 0.2, 0.8],
    "غذا"        => [0.4, 0.7, 0.1],
    "خواب"       => [0.2, 0.3, 0.6],
];

$embeddings = [];
foreach ($tokens as $t) {
    $embeddings[] = $vocab[$t];
}

// ===== ۳) توابع کمکی =====
function dot($a, $b) {
    $s = 0;
    for ($i = 0; $i < count($a); $i++) $s += $a[$i] * $b[$i];
    return $s;
}

function softmax($arr) {
    $max = max($arr);
    $exp = array_map(fn($x) => exp($x - $max), $arr);
    $sum = array_sum($exp);
    return array_map(fn($e) => $e / $sum, $exp);
}

// ===== ۴) Attention (توجه) =====
$n = count($embeddings);
$attention = [];
for ($i = 0; $i < $n; $i++) {
    $scores = [];
    for ($j = 0; $j < $n; $j++) {
        $scores[] = dot($embeddings[$i], $embeddings[$j]);
    }
    $attention[$i] = softmax($scores);
}

echo "=== Attention (توجه) ===\n";
for ($i = 0; $i < $n; $i++) {
    echo "«{$tokens[$i]}» → ";
    for ($j = 0; $j < $n; $j++) {
        printf("%s:%.2f ", $tokens[$j], $attention[$i][$j]);
    }
    echo "\n";
}

// ===== ۵) ترکیب توجه با Embedding =====
// بردار "معنی جمله تا اینجا" (برای آخرین توکن)
$combined = [0, 0, 0];
$last = $n - 1;
for ($i = 0; $i < $n; $i++) {
    for ($k = 0; $k < 3; $k++) {
        $combined[$k] += $attention[$last][$i] * $embeddings[$i][$k];
    }
}

// ===== ۶) Forward از لایه مخفی =====
$W_hidden = [
    [0.3, -0.2, 0.5],   // نرون ۰
    [-0.1, 0.6, 0.2],   // نرون ۱
];
$b_hidden = [0.1, -0.1];

echo "\n=== Forward از لایه مخفی ===\n";
$hidden = [];
for ($j = 0; $j < 2; $j++) {
    $z = $b_hidden[$j];
    for ($k = 0; $k < 3; $k++) $z += $W_hidden[$j][$k] * $combined[$k];
    $hidden[$j] = 1 / (1 + exp(-$z));
    printf("نرون مخفی %d: z=%.3f → sigmoid=%.3f\n", $j, $z, $hidden[$j]);
}

// ===== ۷) لایه خروجی: پیش‌بینی کلمه بعدی =====
$candidates = ["دوست‌دخترم", "ترک", "کرد", "باران", "غذا", "خواب"];
$W_out = [
    [0.8, 0.3],
    [0.5, 0.7],
    [0.2, 0.6],
    [-0.3, 0.1],
    [-0.5, -0.2],
    [-0.1, -0.4],
];

$logits = [];
foreach ($W_out as $w) {
    $z = 0;
    for ($j = 0; $j < 2; $j++) $z += $w[$j] * $hidden[$j];
    $logits[] = $z;
}

$probs = softmax($logits);

echo "\n=== پیش‌بینی کلمه بعدی ===\n";
foreach ($candidates as $i => $c) {
    printf("%-12s → %.1f%%\n", $c, $probs[$i] * 100);
}

$best = array_keys($probs, max($probs))[0];
echo "\nکلمه انتخاب‌شده: «{$candidates[$best]}»\n";