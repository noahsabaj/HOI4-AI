from unittest.mock import patch, call
import executor


class TestCoordinateScaling:
    def test_scales_model_coords_to_window_relative(self):
        window_info = {"window_id": "123", "x": 100, "y": 200, "width": 1920, "height": 1080}
        model_x, model_y = 640, 360  # center of 1280x720

        win_x, win_y = executor.scale_coordinates(
            model_x, model_y, 1280, 720, window_info
        )

        # Window-relative: 640/1280 * 1920 = 960, 360/720 * 1080 = 540
        assert win_x == 960
        assert win_y == 540

    def test_origin_maps_to_zero(self):
        window_info = {"window_id": "123", "x": 50, "y": 75, "width": 1920, "height": 1080}

        win_x, win_y = executor.scale_coordinates(0, 0, 1280, 720, window_info)

        assert win_x == 0
        assert win_y == 0


class TestExecuteClick:
    @patch("subprocess.run")
    def test_calls_xdotool_with_window_relative_coords(self, mock_run):
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}

        executor.execute_click(640, 360, window_info, 1280, 720)

        # Two calls: windowactivate + mousemove/click
        assert mock_run.call_count == 2
        activate_cmd = mock_run.call_args_list[0][0][0]
        assert "windowactivate" in activate_cmd
        click_cmd = mock_run.call_args_list[1][0][0]
        assert "mousemove" in click_cmd
        assert "--window" in click_cmd
        assert "click" in click_cmd


class TestExecuteKey:
    @patch("subprocess.run")
    def test_calls_xdotool_key(self, mock_run):
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}

        executor.execute_key("w", window_info)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "xdotool" in cmd
        assert "key" in cmd


class TestDispatchAction:
    @patch.object(executor, "execute_click")
    def test_dispatches_click_action(self, mock_click):
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}
        action = {"action": "click", "x": 500, "y": 300, "description": "test click"}

        executor.dispatch_action(action, window_info, 1280, 720)

        mock_click.assert_called_once_with(500, 300, window_info, 1280, 720)

    @patch.object(executor, "execute_key")
    def test_dispatches_key_action(self, mock_key):
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}
        action = {"action": "key", "key": "w", "description": "test key"}

        executor.dispatch_action(action, window_info, 1280, 720)

        mock_key.assert_called_once_with("w", window_info)

    def test_done_action_returns_true(self):
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}
        action = {"action": "done", "description": "end cycle"}

        result = executor.dispatch_action(action, window_info, 1280, 720)

        assert result is True
