<?php
// ============================================
// یک نرون ساده برای یادگیری AND + نمایش گرادیان‌ها
// ============================================

$data = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
];

// مقداردهی اولیه
$w1 = (mt_rand() / mt_getrandmax()) * 0.2 - 0.1;
$w2 = (mt_rand() / mt_getrandmax()) * 0.2 - 0.1;
$b  = (mt_rand() / mt_getrandmax()) * 0.2 - 0.1;

function sigmoid(float $z): float {
    return 1.0 / (1.0 + exp(-$z));
}
function sigmoidDerivative(float $a): float {
    return $a * (1.0 - $a);
}

$learningRate = 0.5;
$epochs = 20000;

echo "=== شروع آموزش ===\n";
echo sprintf("وزن اولیه: w1=%.4f, w2=%.4f, b=%.4f\n\n", $w1, $w2, $b);

// ============================================
// حلقه آموزش
// ============================================
for ($epoch = 0; $epoch < $epochs; $epoch++) {

    $totalLoss = 0.0;

    // گرادیان‌های تجمعی این epoch (برای نمایش)
    $sumDw1 = 0.0;
    $sumDw2 = 0.0;
    $sumDb  = 0.0;

    foreach ($data as $row) {
        [$x1, $x2, $target] = $row;

        // Forward
        $z = ($x1 * $w1) + ($x2 * $w2) + $b;
        $a = sigmoid($z);
        $error = $target - $a;
        $totalLoss += $error * $error;

        // Backward
        $dz  = $error * sigmoidDerivative($a);
        $dw1 = $dz * $x1;
        $dw2 = $dz * $x2;
        $db  = $dz;

        // جمع گرادیان‌ها برای نمایش
        $sumDw1 += $dw1;
        $sumDw2 += $dw2;
        $sumDb  += $db;

        // Update
        $w1 += $learningRate * $dw1;
        $w2 += $learningRate * $dw2;
        $b  += $learningRate * $db;
    }

    // ---- نمایش: ۵ epoch اول و ۵ epoch آخر ----
    $showFirst = ($epoch < 5);
    $showLast  = ($epoch >= $epochs - 5);

    if ($showFirst || $showLast) {
        echo sprintf(
            "Epoch %5d | Loss=%.6f | گرادیان‌ها: dw1=%+.4f dw2=%+.4f db=%+.4f | وزن‌ها: w1=%+.4f w2=%+.4f b=%+.4f\n",
            $epoch, $totalLoss, $sumDw1, $sumDw2, $sumDb, $w1, $w2, $b
        );
    }

    // جداکننده بین ابتدا و انتها
    if ($epoch == 4 && $epochs > 10) {
        echo "   ... (وسط آموزش، ۱۹,۹۹۰ epoch دیگر) ...\n";
    }
}

echo "\n=== پایان آموزش ===\n";
echo sprintf("وزن نهایی: w1=%.4f, w2=%.4f, b=%.4f\n\n", $w1, $w2, $b);

// ============================================
// تست
// ============================================
echo "=== نتایج پیش‌بینی ===\n";
echo "x1 | x2 | خروجی خام | پیش‌بینی | انتظار\n";
echo "-----------------------------------------\n";

$correct = 0;
foreach ($data as $row) {
    [$x1, $x2, $target] = $row;
    $z = ($x1 * $w1) + ($x2 * $w2) + $b;
    $a = sigmoid($z);
    $prediction = $a >= 0.5 ? 1 : 0;
    if ($prediction == $target) $correct++;
    echo sprintf(
        " %d |  %d |   %.4f   |    %d     |    %d\n",
        $x1, $x2, $a, $prediction, $target
    );
}

echo "\nدقت: {$correct} از " . count($data) . " نمونه درست پیش‌بینی شد.\n";