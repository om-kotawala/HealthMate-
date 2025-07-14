<?php
session_start();
require 'db.php';

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = $conn->real_escape_string($_POST['username']);
    $password = $_POST['password'];

    $stmt = $conn->prepare("SELECT * FROM admin WHERE username=?");
    $stmt->bind_param("s", $username);
    $stmt->execute();
    $result = $stmt->get_result();
    $admin = $result->fetch_assoc();

    if ($admin && hash_equals($admin['password_hash'], hash('sha256', $password))) {
        $_SESSION['admin'] = $username;
        header("Location: admin_dashboard.php");
    } else {
        file_put_contents("failed_logins.txt", "Failed login attempt by $username on " . date("Y-m-d H:i:s") . "\n", FILE_APPEND);
        echo "<script>alert('Invalid Credentials!'); window.location='admin_login.php';</script>";
    }
}
?>
