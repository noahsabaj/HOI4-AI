import subprocess
import pytest
from unittest.mock import patch, MagicMock
import vision


class TestFindGameWindow:
    def test_returns_window_info_with_frame_extents(self):
        mock_search = MagicMock()
        mock_search.stdout = "12345678\n"
        mock_search.returncode = 0

        mock_geo = MagicMock()
        mock_geo.stdout = "WINDOW=12345678\nX=100\nY=200\nWIDTH=1920\nHEIGHT=1080\nSCREEN=0\n"
        mock_geo.returncode = 0

        mock_xprop = MagicMock()
        mock_xprop.stdout = "_NET_FRAME_EXTENTS(CARDINAL) = 0, 0, 30, 0\n"
        mock_xprop.returncode = 0

        with patch("subprocess.run", side_effect=[mock_search, mock_geo, mock_xprop]):
            info = vision.find_game_window("Hearts of Iron")

        assert info["window_id"] == "12345678"
        # X unchanged (no left frame), Y adjusted by title bar (30px)
        assert info["x"] == 100
        assert info["y"] == 230
        # Height reduced by title bar
        assert info["width"] == 1920
        assert info["height"] == 1050

    def test_returns_none_when_not_found(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            info = vision.find_game_window("Hearts of Iron")

        assert info is None


class TestCaptureScreenshot:
    def test_returns_base64_string(self):
        with patch.object(vision, "mss") as mock_mss_mod:
            mock_sct = MagicMock()
            mock_pixel_data = MagicMock()
            mock_pixel_data.bgra = b"\x00" * (100 * 100 * 4)  # BGRA = 4 bytes per pixel
            mock_pixel_data.size = (100, 100)
            mock_pixel_data.width = 100
            mock_pixel_data.height = 100
            mock_sct.grab.return_value = mock_pixel_data
            mock_mss_mod.mss.return_value.__enter__ = MagicMock(return_value=mock_sct)
            mock_mss_mod.mss.return_value.__exit__ = MagicMock(return_value=False)

            window_info = {"x": 0, "y": 0, "width": 100, "height": 100}
            result = vision.capture_screenshot(window_info, 1280, 720)

        assert isinstance(result["base64"], str)
        assert len(result["base64"]) > 0
        assert result["width"] == 1280
        assert result["height"] == 720
